#!/usr/bin/env python3
"""Authenticated, single-session signaling for the existing cloud GPU worker.

Default bind is loopback. Public HTTPS exposure and TURN provisioning are separate
operator steps. No tunnel, paid service, camera, or microphone is started here.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import io
import logging
import os
import re
from pathlib import Path
import secrets
import time

from aiohttp import web
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from aiortc.sdp import SessionDescription

from headless_webrtc_track import ProcessedVideoTrack

LOG = logging.getLogger('headless-signaling')
MANAGER = web.AppKey('manager', object)


def decode_identity(data):
    """Bound image dimensions before pixel decoding; never save the upload."""
    import warnings
    import numpy as np
    from PIL import Image, ImageOps, UnidentifiedImageError
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in ('JPEG', 'PNG') or getattr(image, 'n_frames', 1) != 1:
                    raise ValueError('Choose a single JPEG or PNG image.')
                width, height = image.size
                if min(width, height) < 64 or max(width, height) > 4096 or width * height > 8_000_000:
                    raise ValueError('Use an image from 64 pixels up to 4096 pixels, at most 8 megapixels.')
                image = ImageOps.exif_transpose(image).convert('RGB')
                image.thumbnail((1024, 1024))
                return np.asarray(image)[:, :, ::-1].copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError('Cannot decode this photo. Choose a valid JPEG or PNG.') from error


async def read_limited(request, maximum):
    data = bytearray()
    async for chunk in request.content.iter_chunked(65536):
        data.extend(chunk)
        if len(data) > maximum:
            raise web.HTTPRequestEntityTooLarge(max_size=maximum, actual_size=len(data))
    return bytes(data)


def validate_ice_servers(value):
    if not isinstance(value, list) or len(value) > 10:
        raise ValueError('ICE configuration must be a list of at most ten servers.')
    result = []
    for server in value:
        if not isinstance(server, dict) or set(server) - {'urls', 'username', 'credential'}:
            raise ValueError('Invalid ICE server fields.')
        urls = server.get('urls')
        if isinstance(urls, str):
            urls = [urls]
        if not isinstance(urls, list) or not urls or len(urls) > 10:
            raise ValueError('ICE server urls must be a nonempty string or list.')
        if not all(isinstance(url, str) and len(url) < 2048 and
                   url.startswith(('stun:', 'stuns:', 'turn:', 'turns:')) for url in urls):
            raise ValueError('Invalid STUN/TURN URL.')
        clean = {'urls': urls}
        for field in ('username', 'credential'):
            if field in server:
                if not isinstance(server[field], str) or len(server[field]) > 4096:
                    raise ValueError('Invalid TURN credential.')
                clean[field] = server[field]
        if any(url.startswith(('turn:', 'turns:')) for url in urls):
            if not clean.get('username') or not clean.get('credential'):
                raise ValueError('TURN needs a username and credential.')
        result.append(clean)
    return result


class SessionManager:
    def __init__(self, worker, ice_servers, session_seconds=600, idle_seconds=30):
        self.worker = worker
        self.configuration = RTCConfiguration(iceServers=[RTCIceServer(**s) for s in ice_servers])
        self.session = None
        self.lock = asyncio.Lock()
        self.session_seconds = session_seconds
        self.idle_seconds = idle_seconds
        self.watchdog = None
        self.stopping = False
        self.closed_ids = {}
        self.identity_revision = 0

    async def set_identity(self, data):
        async with self.lock:
            if self.stopping:
                raise web.HTTPServiceUnavailable(text='Server is shutting down.')
            if self.session is not None:
                raise web.HTTPConflict(text='Stop the video before changing the face photo.')
            def update():
                image = decode_identity(data)
                self.worker.set_identity(image)
            operation = asyncio.create_task(asyncio.to_thread(update))
            try:
                await asyncio.shield(operation)
            except asyncio.CancelledError:
                await operation  # Do not release admission while GPU work is running.
                self.identity_revision += 1
                raise
            self.identity_revision += 1
            return {'identityRevision': self.identity_revision}

    def _purge_closed_ids(self):
        now = time.monotonic()
        self.closed_ids = {key: expiry for key, expiry in self.closed_ids.items() if expiry > now}

    async def open(self, sdp, session_id=None, swap_enabled=True, preserve_mouth=False):
        async with self.lock:
            if self.stopping:
                raise web.HTTPServiceUnavailable(text='Server is shutting down.')
            self._purge_closed_ids()
            if session_id in self.closed_ids:
                raise web.HTTPGone(text='This connection attempt was stopped. Start a new session.')
            if self.session is not None:
                raise web.HTTPConflict(text='The worker already has an active session. Stop it first.')
            if getattr(self.worker,'mode','face') in ('portrait','avatar') and preserve_mouth:
                raise ValueError('Real-mouth overlay is unavailable in avatar modes.')
            if getattr(self.worker,'mode','face') == 'avatar' and not swap_enabled:
                raise ValueError('Full-body avatar mode does not return the source camera.')
            if hasattr(self.worker,'reset_motion'):
                self.worker.reset_motion()
            pc = RTCPeerConnection(self.configuration)
            session = {'id': session_id or secrets.token_urlsafe(24), 'pc': pc, 'track': None,
                       'created': time.monotonic(), 'activity': time.monotonic(),
                       'captured': 0, 'closing': False}
            self.session = session

            @pc.on('track')
            def on_track(track):
                if track.kind != 'video' or session['track'] is not None:
                    track.stop()
                    return
                processed = ProcessedVideoTrack(track, self.worker, swap_enabled, preserve_mouth)
                session['track'] = processed
                pc.addTrack(processed)

            @pc.on('connectionstatechange')
            async def on_state():
                if pc.connectionState in ('failed', 'closed'):
                    await self.close(session['id'])

            try:
                async def negotiate():
                    await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type='offer'))
                    if session['track'] is None:
                        raise ValueError('No incoming video track was offered.')
                    await pc.setLocalDescription(await pc.createAnswer())
                # CPU tracking runs on Python 3.10; asyncio.timeout is 3.11+.
                # wait_for also waits for cancellation before admission unlocks.
                await asyncio.wait_for(negotiate(),timeout=25)
                return {'id': session['id'], 'sdp': pc.localDescription.sdp,
                        'type': pc.localDescription.type}
            except BaseException:
                # Keep admission locked until all prior inference is drained.
                await self._close_locked(session)
                raise

    async def _close_locked(self, session):
        if session['closing']:
            return
        session['closing'] = True
        try:
            try:
                await session['pc'].close()
            finally:
                if session['track'] is not None:
                    await session['track'].aclose()
        finally:
            if self.session is session:
                self.session = None

    async def close(self, session_id):
        async with self.lock:
            # Stop may arrive before its HTTP offer. Remember bounded tombstones
            # so a delayed offer cannot recreate an already-stopped session.
            self._purge_closed_ids()
            self.closed_ids[session_id] = time.monotonic() + 60
            if len(self.closed_ids) > 128:
                self.closed_ids.pop(next(iter(self.closed_ids)))
            if self.session is not None and secrets.compare_digest(self.session['id'].encode(), session_id.encode()):
                await self._close_locked(self.session)

    def status(self, session_id):
        session = self.session
        if session is None or not secrets.compare_digest(session['id'].encode(), session_id.encode()):
            raise web.HTTPNotFound(text='Session ended or does not exist.')
        track = session['track']
        return {'state': session['pc'].connectionState, 'iceState': session['pc'].iceConnectionState,
                'processed': track.processed if track else 0, 'dropped': track.dropped if track else 0,
                'error': bool(track and track.error), 'expiresIn': max(0, int(
                    self.session_seconds - (time.monotonic() - session['created']))),
                'faceStatus': track.last_status if track else 'waiting',
                'swappedFrames': track.swapped_frames if track else 0,
                'avatarFrames': track.avatar_frames if track else 0,
                'noFaceFrames': track.no_face_frames if track else 0}

    async def watch(self):
        while True:
            await asyncio.sleep(1)
            session = self.session
            if session is None:
                continue
            now = time.monotonic()
            track = session['track']
            if track and track.captured != session['captured']:
                session['captured'] = track.captured
                session['activity'] = now
            if (now - session['created'] > self.session_seconds or
                    now - session['activity'] > self.idle_seconds or
                    (track and track.error is not None)):
                try:
                    await self.close(session['id'])
                except Exception:
                    LOG.exception('Session cleanup failed')


def create_app(worker, token, ice_servers=None, allowed_origins=(), relay_only=False,
               session_seconds=600, idle_seconds=30):
    if not isinstance(token, str) or len(token) < 32:
        raise ValueError('Use a signaling token of at least 32 characters.')
    ice_servers = validate_ice_servers(ice_servers or [])
    if relay_only and not any(url.startswith(('turn:', 'turns:'))
                              for s in ice_servers for url in s['urls']):
        raise ValueError('Relay-only mode requires a configured TURN server.')
    manager = SessionManager(worker, ice_servers, session_seconds, idle_seconds)
    allowed_origins = set(allowed_origins)

    @web.middleware
    async def protect(request, handler):
        if request.path != '/' and request.path not in ('/client.js', '/client.css', '/demo.png',
                                                       '/output', '/output.js', '/output.css'):
            supplied = request.headers.get('Authorization', '')
            if not secrets.compare_digest(supplied.encode(), ('Bearer ' + token).encode()):
                raise web.HTTPUnauthorized(text='A valid connection token is required.')
            origin = request.headers.get('Origin')
            same_origin = f'{request.scheme}://{request.host}'
            if origin and origin != same_origin and origin not in allowed_origins:
                raise web.HTTPForbidden(text='Origin is not allowed.')
        response = await handler(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; "
            "img-src 'self' data:; media-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'")
        response.headers['Permissions-Policy'] = 'camera=(self), microphone=()'
        return response

    app = web.Application(middlewares=[protect], client_max_size=4 * 1024 * 1024)
    app[MANAGER] = manager

    async def asset(request):
        files = {'/': ('webrtc_client.html', 'text/html'),
                 '/client.js': ('webrtc_client.js', 'application/javascript'),
                 '/client.css': ('webrtc_client.css', 'text/css'),
                 '/output': ('obs_output.html', 'text/html'),
                 '/output.js': ('obs_output.js', 'application/javascript'),
                 '/output.css': ('obs_output.css', 'text/css')}
        filename, mime = files[request.path]
        return web.Response(text=(Path(__file__).parent / filename).read_text(), content_type=mime)

    async def demo(request):
        path = Path(__file__).resolve().parents[1] / 'doc/celebs/Millie_Park/Millie_Park.png'
        return web.FileResponse(path)

    async def config(request):
        return web.json_response({'iceServers': ice_servers, 'relayOnly': relay_only,
                                  'sessionSeconds': session_seconds, 'hasIceServers': bool(ice_servers),
                                  'renderingMode':getattr(worker,'mode','face'),
                                  'portraitEngine':getattr(worker,'engine',None),
                                  'identityRevision': manager.identity_revision})

    async def identity(request):
        if request.content_type not in ('image/jpeg', 'image/png'):
            raise web.HTTPUnsupportedMediaType(text='Choose a JPEG or PNG photo.')
        try:
            return web.json_response(await manager.set_identity(await read_limited(request, 4 * 1024 * 1024)))
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from error

    async def offer(request):
        try:
            data = json.loads(await read_limited(request, 65536))
            if (not isinstance(data, dict) or set(data) - {'sdp', 'type', 'id', 'swapEnabled', 'preserveMouth'} or not {'sdp', 'type'} <= set(data) or
                    data['type'] != 'offer' or not isinstance(data['sdp'], str)):
                raise ValueError('Expected a video SDP offer.')
            if any(type(data[field]) is not bool for field in ('swapEnabled', 'preserveMouth') if field in data):
                raise ValueError('Video options must be booleans.')
            if 'id' in data and (not isinstance(data['id'], str) or
                    re.fullmatch(r'[A-Za-z0-9_-]{16,64}', data['id']) is None):
                raise ValueError('Invalid session id.')
            parsed = SessionDescription.parse(data['sdp'])
            if (len(parsed.media) != 1 or parsed.media[0].kind != 'video' or
                    parsed.media[0].direction != 'sendrecv'):
                raise ValueError('Offer exactly one outgoing video track and no audio/data tracks.')
            answer = await manager.open(data['sdp'], data.get('id'),
                                        data.get('swapEnabled', True), data.get('preserveMouth', False))
            return web.json_response(answer)
        except web.HTTPException:
            raise
        except (ValueError, KeyError, IndexError, TypeError, AssertionError):
            raise web.HTTPBadRequest(text='Invalid video offer.')
        except (TimeoutError, asyncio.TimeoutError):
            raise web.HTTPGatewayTimeout(text='ICE negotiation timed out. Check STUN/TURN configuration.')

    async def status(request):
        return web.json_response(manager.status(request.match_info['id']))

    async def stop(request):
        await manager.close(request.match_info['id'])
        return web.Response(status=204)

    app.router.add_get('/', asset)
    app.router.add_get('/client.js', asset)
    app.router.add_get('/client.css', asset)
    app.router.add_get('/output', asset)
    app.router.add_get('/output.js', asset)
    app.router.add_get('/output.css', asset)
    app.router.add_get('/demo.png', demo)
    app.router.add_get('/config', config)
    app.router.add_post('/identity', identity)
    app.router.add_post('/offer', offer)
    app.router.add_get('/session/{id}', status)
    app.router.add_delete('/session/{id}', stop)

    async def startup(app):
        manager.watchdog = asyncio.create_task(manager.watch())

    async def cleanup(app):
        manager.stopping = True
        manager.watchdog.cancel()
        await asyncio.gather(manager.watchdog, return_exceptions=True)
        async with manager.lock:
            try:
                if manager.session:
                    await manager._close_locked(manager.session)
            finally:
                await asyncio.to_thread(worker.close)

    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)
    return app


def load_token(path):
    path = Path(path)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        token = path.read_text().strip()
    else:
        token = secrets.token_urlsafe(32)
        with os.fdopen(fd, 'w') as handle:
            handle.write(token + '\n')
    if len(token) < 32:
        raise ValueError('Connection token file is invalid.')
    return token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-image')
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--token-file', default='/tmp/dfl-signaling-token')
    parser.add_argument('--ice-config', help='JSON file containing an iceServers list.')
    parser.add_argument('--allowed-origin', action='append', default=[])
    parser.add_argument('--relay-only', action='store_true')
    parser.add_argument('--diagnostic-cpu', action='store_true', help='Color inversion test; no face swap or GPU claim.')
    parser.add_argument('--mode', choices=['face','portrait','avatar'], default='face')
    parser.add_argument('--avatar-config', type=Path, help='Private prepared LHM++/tracker configuration JSON.')
    parser.add_argument('--gpu-python', help='Prepared isolated LHM++ Python executable.')
    parser.add_argument('--portrait-runtime', help='Pinned runtime directory for the optional LivePortrait engine.')
    parser.add_argument('--portrait-engine',choices=['lia','liveportrait'],default='lia')
    parser.add_argument('--evidence-dir', default='/kaggle/working/dfl-signaling-evidence')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    token = os.environ.get('DFL_SIGNALING_TOKEN') or load_token(args.token_file)
    if len(token) < 32:
        raise ValueError('Connection token must contain at least 32 characters.')
    ice_servers = validate_ice_servers(json.loads(Path(args.ice_config).read_text()) if args.ice_config else [])
    if args.mode in ('portrait','avatar') and args.diagnostic_cpu:
        parser.error('Avatar modes require GPU; diagnostic CPU mode cannot enable them.')
    if args.mode == 'avatar':
        if not args.avatar_config or not args.gpu_python:
            parser.error('Avatar mode needs --avatar-config and --gpu-python.')
        from avatar_stream_tracker import MediaPipeAvatarTracker
        from avatar_pipe_client import AvatarPipeClient
        from headless_lhmpp_stream_worker import HeadlessLHMStreamWorker
        config = json.loads(args.avatar_config.read_text())
        Path(args.evidence_dir).mkdir(parents=True,exist_ok=True)
        tracker = MediaPipeAvatarTracker(config)
        try:
            renderer = AvatarPipeClient(args.gpu_python,args.avatar_config,
                                        Path(args.evidence_dir)/'avatar-gpu.log')
        except BaseException:
            tracker.close();raise
        worker = HeadlessLHMStreamWorker(tracker,renderer)
    elif args.mode == 'portrait':
        if not args.source_image:
            parser.error('Portrait mode needs --source-image.')
        if args.portrait_engine == 'liveportrait':
            if not args.portrait_runtime:
                parser.error('LivePortrait needs --portrait-runtime.')
            from headless_portrait_worker import HeadlessPortraitWorker as PortraitWorker
        else:
            from headless_lia_portrait_worker import HeadlessLiaPortraitWorker as PortraitWorker
        import cv2
        worker = PortraitWorker(args.source_image,args.evidence_dir,args.portrait_runtime)
        worker.process_frame(cv2.imread(args.source_image))
    elif args.diagnostic_cpu:
        class DiagnosticWorker:
            def set_identity(self, image):
                pass  # Explicit CPU UI test only; never claims identity inference.
            def process_frame(self, frame):
                return 255 - frame, 'diagnostic'
            def close(self):
                pass
        worker = DiagnosticWorker()
    else:
        if not args.source_image:
            parser.error('--source-image is required for GPU face swapping')
        from headless_cloud_worker import HeadlessCloudWorker
        import cv2
        worker = HeadlessCloudWorker(args.source_image, args.evidence_dir, require_cuda=True)
        image = cv2.imread(args.source_image)
        worker.process_frame(image)  # Warm the complete CUDA path before accepting video.
    app = create_app(worker, token, ice_servers, args.allowed_origin, args.relay_only)
    LOG.info('Token source: %s (value not printed). Diagnostic CPU mode: %s',
             'DFL_SIGNALING_TOKEN environment variable' if os.environ.get('DFL_SIGNALING_TOKEN') else args.token_file,
             args.diagnostic_cpu)
    web.run_app(app, host=args.bind, port=args.port, access_log=None)


if __name__ == '__main__':
    main()

