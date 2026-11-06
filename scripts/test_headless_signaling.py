"""Real HTTP checks for signaling authorization and session lifecycle."""
import asyncio
import json
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from aiohttp.test_utils import AioHTTPTestCase
from aiortc import VideoStreamTrack

import headless_webrtc_server as server

TOKEN = 'test-only-' + 'x' * 40
HEADERS = {'Authorization': 'Bearer ' + TOKEN}
OFFER = ('v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n'
         'm=video 9 UDP/TLS/RTP/SAVPF 96\r\nc=IN IP4 0.0.0.0\r\n'
         'a=sendrecv\r\na=rtpmap:96 VP8/90000\r\n')


class FakeWorker:
    def set_identity(self, image):
        self.identity_shape = image.shape
    def close(self):
        self.closed = True


class FakePeer:
    connectionState = 'new'
    iceConnectionState = 'new'
    gate = None
    def __init__(self, configuration):
        self.handlers = {}
    def on(self, event):
        def register(callback):
            self.handlers[event] = callback
            return callback
        return register
    async def setRemoteDescription(self, description):
        if self.gate:
            await self.gate.wait()
        source = VideoStreamTrack()
        source.kind = 'video'
        self.handlers['track'](source)
    def addTrack(self, track):
        pass
    async def createAnswer(self):
        return SimpleNamespace(sdp='answer', type='answer')
    async def setLocalDescription(self, description):
        self.localDescription = description
        self.connectionState = self.iceConnectionState = 'connected'
    async def close(self):
        self.connectionState = self.iceConnectionState = 'closed'


class SignalingTests(AioHTTPTestCase):
    async def get_application(self):
        self.worker = FakeWorker()
        return server.create_app(self.worker, TOKEN, allowed_origins=['https://approved.invalid'], idle_seconds=.05)

    async def test_assets_do_not_expose_the_token(self):
        for path in ['/', '/client.js', '/client.css']:
            response = await self.client.get(path)
            self.assertEqual(response.status, 200)
            self.assertNotIn(TOKEN, await response.text())
            self.assertEqual(response.headers['Cache-Control'], 'no-store')
            self.assertIn("frame-ancestors 'none'", response.headers['Content-Security-Policy'])

    async def test_config_requires_token_and_rejects_untrusted_origin(self):
        response = await self.client.get('/config')
        self.assertEqual(response.status, 401)
        response = await self.client.get('/config', headers={**HEADERS, 'Origin': 'https://untrusted.invalid'})
        self.assertEqual(response.status, 403)
        response = await self.client.get('/config', headers={**HEADERS, 'Origin':'https://approved.invalid'})
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())['iceServers'], [])
        self.assertEqual(response.headers['Cache-Control'], 'no-store')

    async def test_invalid_offers_are_rejected_before_session_creation(self):
        cases = [[], {'type':'answer','sdp':OFFER}, {'type':'offer','sdp':3},
                 {'type':'offer','sdp':'broken'}, {'type':'offer','sdp':OFFER,'id':'../bad'},
                 {'type':'offer','sdp':OFFER.replace('m=video','m=audio')},
                 {'type':'offer','sdp':OFFER.replace('sendrecv','recvonly')}]
        for data in cases:
            response = await self.client.post('/offer', json=data, headers=HEADERS)
            self.assertEqual(response.status, 400, str(data))
            self.assertIsNone(self.app[server.MANAGER].session)
        response = await self.client.post('/offer', data='bad-json', headers=HEADERS)
        self.assertEqual(response.status, 400)

    async def test_oversized_offer_is_rejected(self):
        response = await self.client.post('/offer', json={'sdp':'x'*70000,'type':'offer'}, headers=HEADERS)
        self.assertEqual(response.status, 413)

    async def test_identity_upload_auth_validation_and_active_session(self):
        from PIL import Image
        buffer = io.BytesIO()
        Image.new('RGB', (128, 128), (100, 60, 20)).save(buffer, format='PNG')
        data = buffer.getvalue()
        headers = {**HEADERS, 'Content-Type':'image/png'}
        response = await self.client.post('/identity', data=data)
        self.assertEqual(response.status, 401)
        response = await self.client.post('/identity', data=b'broken', headers=headers)
        self.assertEqual(response.status, 400)
        self.assertEqual(self.app[server.MANAGER].identity_revision, 0)
        response = await self.client.post('/identity', data=data, headers=headers)
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())['identityRevision'], 1)
        self.assertEqual(self.worker.identity_shape, (128,128,3))
        with patch.object(server, 'RTCPeerConnection', FakePeer):
            response = await self.client.post('/offer', json={'sdp':OFFER,'type':'offer'}, headers=HEADERS)
            self.assertEqual(response.status, 200)
            response = await self.client.post('/identity', data=data, headers=headers)
            self.assertEqual(response.status, 409)
        response = await self.client.post('/identity', data=b'x'*(4*1024*1024+1), headers=headers)
        self.assertEqual(response.status, 413)

    async def test_invalid_video_options_are_rejected(self):
        for field in ('swapEnabled', 'preserveMouth'):
            response = await self.client.post('/offer', json={'sdp':OFFER,'type':'offer',field:'yes'}, headers=HEADERS)
            self.assertEqual(response.status,400)

    async def test_single_slot_status_stop_and_readmission(self):
        with patch.object(server, 'RTCPeerConnection', FakePeer):
            data = {'sdp':OFFER,'type':'offer','id':'client-test-session-001'}
            response = await self.client.post('/offer', json=data, headers=HEADERS)
            self.assertEqual(response.status, 200, await response.text())
            answer = await response.json()
            self.assertEqual(answer['id'], data['id'])
            response = await self.client.post('/offer', json=data, headers=HEADERS)
            self.assertEqual(response.status, 409)
            response = await self.client.get('/session/'+answer['id'], headers=HEADERS)
            self.assertEqual((await response.json())['state'], 'connected')
            response = await self.client.delete('/session/'+answer['id'])
            self.assertEqual(response.status, 401)
            response = await self.client.delete('/session/'+answer['id'], headers=HEADERS)
            self.assertEqual(response.status, 204)
            self.assertIsNone(self.app[server.MANAGER].session)
            data['id'] = 'client-test-session-002'
            response = await self.client.post('/offer', json=data, headers=HEADERS)
            self.assertEqual(response.status, 200)

    async def test_stop_before_offer_prevents_a_late_orphan_session(self):
        with patch.object(server, 'RTCPeerConnection', FakePeer):
            sid = 'client-stopped-before-offer'
            response = await self.client.delete('/session/'+sid, headers=HEADERS)
            self.assertEqual(response.status, 204)
            response = await self.client.post('/offer', json={'sdp':OFFER,'type':'offer','id':sid}, headers=HEADERS)
            self.assertEqual(response.status, 410)
            self.assertIsNone(self.app[server.MANAGER].session)
            for index in range(150):
                await self.app[server.MANAGER].close(f'stopped-session-{index}')
            self.assertLessEqual(len(self.app[server.MANAGER].closed_ids),128)

    async def test_idle_session_releases_slot(self):
        with patch.object(server, 'RTCPeerConnection', FakePeer):
            response = await self.client.post('/offer', json={'sdp':OFFER,'type':'offer'}, headers=HEADERS)
            self.assertEqual(response.status, 200)
            async with asyncio.timeout(3):
                while self.app[server.MANAGER].session is not None:
                    await asyncio.sleep(.02)


class ConfigTests(unittest.TestCase):
    def test_turn_requires_credentials_and_relay_requires_turn(self):
        with self.assertRaises(ValueError):
            server.validate_ice_servers([{'urls':'turn:relay.invalid:3478'}])
        with self.assertRaises(ValueError):
            server.create_app(FakeWorker(), TOKEN, relay_only=True)
        with self.assertRaises(ValueError):
            server.create_app(FakeWorker(), 'short')
        with self.assertRaises(ValueError):
            server.validate_ice_servers([{'urls':'https://invalid.example'}])


if __name__ == '__main__':
    unittest.main()
