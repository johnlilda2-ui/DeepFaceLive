#!/usr/bin/env python3
"""Actual HTTP signaling and WebRTC video through the existing worker.

Runs entirely on loopback. GPU is required by default. --transport-only uses CPU
color inversion and cannot claim GPU or external Windows verification.
"""
import argparse
import asyncio
import json
from pathlib import Path
import secrets
import tempfile

import cv2
import numpy as np
from aiohttp import ClientSession, web
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription

from headless_webrtc_server import create_app, MANAGER, validate_ice_servers
from test_webrtc_gpu_roundtrip import SampleVideo


class DiagnosticWorker:
    def process_frame(self, frame):
        return 255 - frame, 'diagnostic'
    def close(self):
        pass


def relay_only_sdp(sdp):
    """Test helper: advertise only relay candidates; never log SDP or credentials."""
    lines = sdp.splitlines()
    relays = [line for line in lines if line.startswith('a=candidate:') and ' typ relay' in line]
    if not relays:
        raise RuntimeError('TURN returned no relay candidates; relay test cannot pass.')
    return '\r\n'.join(line for line in lines if not line.startswith('a=candidate:') or line in relays) + '\r\n'


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport-only', action='store_true')
    parser.add_argument('--ice-config', help='Private ICE JSON file; never included in reports.')
    parser.add_argument('--force-relay', action='store_true', help='Exchange only relay candidates and verify actual video.')
    parser.add_argument('--exercise-upload-mouth', action='store_true', help='GPU identity upload and mouth-preserving return test.')
    parser.add_argument('--portrait-runtime',help='Test the separate portrait GPU worker instead of face swapping.')
    parser.add_argument('--portrait-engine',choices=['lia','liveportrait'])
    parser.add_argument('--source-image')
    parser.add_argument('--target-image')
    parser.add_argument('--output-dir', default='artifacts/http-signaling')
    args = parser.parse_args()
    ice_servers = validate_ice_servers(json.loads(Path(args.ice_config).read_text()) if args.ice_config else [])
    if args.force_relay and not any(u.startswith(('turn:', 'turns:')) for s in ice_servers for u in s['urls']):
        parser.error('--force-relay requires a TURN server in --ice-config')
    portrait_engine = args.portrait_engine or ('liveportrait' if args.portrait_runtime else None)
    if args.transport_only and args.exercise_upload_mouth:
        parser.error('--exercise-upload-mouth requires the GPU worker')
    if portrait_engine and (args.transport_only or args.exercise_upload_mouth):
        parser.error('Portrait mode requires GPU and does not use the face-swap mouth overlay.')
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='run-', dir=root))
    if args.transport_only:
        worker = DiagnosticWorker()
        target = np.full((128, 128, 3), 40, np.uint8)
    else:
        if not args.source_image or not args.target_image:
            parser.error('GPU mode requires source and target images')
        if portrait_engine == 'liveportrait':
            from headless_portrait_worker import HeadlessPortraitWorker
            worker = HeadlessPortraitWorker(args.source_image,run,args.portrait_runtime)
        elif portrait_engine == 'lia':
            from headless_lia_portrait_worker import HeadlessLiaPortraitWorker
            worker = HeadlessLiaPortraitWorker(args.source_image,run)
        else:
            from headless_cloud_worker import HeadlessCloudWorker
            worker = HeadlessCloudWorker(args.source_image, run, require_cuda=True)
        target = cv2.imread(args.target_image)
        if target is None:
            raise ValueError('Cannot decode target image')
    expected, _ = await asyncio.to_thread(worker.process_frame, target)
    token = secrets.token_urlsafe(32)  # Never logged or written into artifacts.
    headers = {'Authorization':'Bearer '+token}
    app = create_app(worker, token, ice_servers=ice_servers, relay_only=args.force_relay, idle_seconds=60)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    url = f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'
    peer = RTCPeerConnection(RTCConfiguration(iceServers=[RTCIceServer(**s) for s in ice_servers]))
    source = SampleVideo(target)
    peer.addTrack(source)
    returned = asyncio.get_running_loop().create_future()
    @peer.on('track')
    def on_track(track):
        if not returned.done():
            returned.set_result(track)
    try:
        async with ClientSession() as http:
            response = await http.get(url+'/config')
            assert response.status == 401, 'Unauthenticated configuration was exposed'
            upload_change = mouth_change = None
            if portrait_engine:
                photo_headers = {**headers, "Content-Type":"image/png"}
                response = await http.post(url+"/identity", headers=photo_headers, data=Path(args.target_image).read_bytes())
                assert response.status == 200, await response.text()
                other, _ = await asyncio.to_thread(worker.process_frame, target)
                upload_change = float(np.abs(other.astype(float)-expected).mean())
                assert upload_change > .1, "Portrait upload did not change avatar appearance"
                response = await http.post(url+"/identity", headers=photo_headers, data=Path(args.source_image).read_bytes())
                assert response.status == 200, await response.text()
                expected, _ = await asyncio.to_thread(worker.process_frame, target)
            if args.exercise_upload_mouth:
                photo_headers = {**headers, 'Content-Type':'image/png'}
                response = await http.post(url+'/identity',headers=photo_headers,data=Path(args.target_image).read_bytes())
                assert response.status == 200, await response.text()
                other_identity, _ = await asyncio.to_thread(worker.process_frame,target)
                upload_change = float(np.abs(other_identity.astype(float)-expected).mean())
                assert upload_change > .1, 'Changing the uploaded identity did not change output'
                response = await http.post(url+'/identity',headers=photo_headers,data=Path(args.source_image).read_bytes())
                assert response.status == 200, await response.text()
                ok, jpeg = cv2.imencode('.jpg', cv2.imread(args.source_image))
                assert ok
                response = await http.post(url+'/identity',headers={**headers,'Content-Type':'image/jpeg'},data=jpeg.tobytes())
                assert response.status == 200, await response.text()
                expected, _ = await asyncio.to_thread(worker.process_frame,target)
                with_mouth, _ = await asyncio.to_thread(worker.process_frame,target,preserve_mouth=True)
                mouth_change = float(np.abs(with_mouth.astype(float)-expected).mean())
                assert mouth_change > .01, 'Mouth preservation did not change output'
                expected = with_mouth
            await peer.setLocalDescription(await peer.createOffer())
            sid = secrets.token_urlsafe(24)
            response = await http.post(url+'/offer', headers=headers, json={
                'type':'offer','sdp':relay_only_sdp(peer.localDescription.sdp) if args.force_relay else peer.localDescription.sdp,'id':sid,
                'preserveMouth':args.exercise_upload_mouth})
            if response.status != 200:
                raise RuntimeError(f'Signaling failed ({response.status}): {await response.text()}')
            answer = await response.json()
            if args.force_relay:
                answer['sdp'] = relay_only_sdp(answer['sdp'])
            await peer.setRemoteDescription(RTCSessionDescription(sdp=answer['sdp'],type=answer['type']))
            remote = await asyncio.wait_for(returned, 15)
            timestamps = []
            for _ in range(5):
                frame = await asyncio.wait_for(remote.recv(), 30)
                timestamps.append(float(frame.pts * frame.time_base))
                received = frame.to_ndarray(format='bgr24')
                assert received.shape == expected.shape, 'Unexpected return canvas dimensions'
            assert all(b > a for a,b in zip(timestamps,timestamps[1:])), 'Nonmonotonic timestamps'
            comparison = cv2.resize(target,(received.shape[1],received.shape[0]))
            original_error = float(np.abs(received.astype(float)-comparison).mean())
            processed_error = float(np.abs(received.astype(float)-expected).mean())
            assert processed_error+1 < original_error, 'Returned video did not match processing'
            response = await http.get(url+'/session/'+sid, headers=headers)
            assert response.status == 200
            status = await response.json()
            assert status['state'] == 'connected' and status['processed'] >= 5 and not status['error'], status
            stats = await peer.getStats()
            packets = sum(x.packetsReceived for x in stats.values() if x.type == 'inbound-rtp')
            assert packets > 0, 'No incoming RTP packets'
            response = await http.delete(url+'/session/'+sid, headers=headers)
            assert response.status == 204
            assert app[MANAGER].session is None, 'Session slot not released'
            response = await http.get(url+'/session/'+sid, headers=headers)
            assert response.status == 404
            report = {'scope':'loopback HTTP signaling and WebRTC video', 'frames_received':5,
                      'relay_only_candidate_exchange':args.force_relay,
                      'rendering_mode':'portrait' if portrait_engine else 'face',
                      'portrait_engine':portrait_engine,
                      'packets_received':packets,'output_shape':list(received.shape),
                      'original_mae':original_error,'processed_reference_mae':processed_error,
                      'session':status,'stop_released_session':True,'gpu_verified':not args.transport_only,
                      'identity_upload_verified':bool(args.exercise_upload_mouth or portrait_engine),
                      'mouth_preservation_verified':args.exercise_upload_mouth,
                      'identity_change_mae':upload_change,'mouth_change_mae':mouth_change,
                      'external_windows_tested':False}
            assert cv2.imwrite(str(run/'returned_frame.png'),received)
    finally:
        await peer.close()
        source.stop()
        await runner.cleanup()  # GPU profiles must pass before success is recorded.
    if not args.transport_only:
        report['execution_evidence'] = str(worker.profile_dir/('portrait-evidence.json' if portrait_engine else 'execution_evidence.json'))
    report['status'] = 'passed'
    (run/'signaling-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    print('HTTP SIGNALING GPU TEST PASSED' if not args.transport_only else 'HTTP SIGNALING CPU TRANSPORT TEST PASSED',run,flush=True)


if __name__ == '__main__':
    asyncio.run(main())
