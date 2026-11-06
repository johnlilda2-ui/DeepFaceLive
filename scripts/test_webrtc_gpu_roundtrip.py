#!/usr/bin/env python3
"""Verify real WebRTC RTP/DTLS video round-trip through the frame worker.

Both peers run on one host, without external ICE servers. GPU mode uses the
fictional sample images. --transport-only tests transport with a CPU color
transform and cannot claim GPU verification. No paid hosting is started.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from aiortc import RTCConfiguration, RTCPeerConnection, VideoStreamTrack
from av import VideoFrame

from headless_webrtc_track import ProcessedVideoTrack


class SampleVideo(VideoStreamTrack):
    def __init__(self, image):
        super().__init__()
        self.image = image

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = VideoFrame.from_ndarray(self.image, format='bgr24')
        frame.pts, frame.time_base = pts, time_base
        return frame


class TransportTestWorker:
    def process_frame(self, image):
        time.sleep(.06)  # Deliberately slower than 30 fps to exercise backpressure.
        return 255 - image, 'transport_test'


async def roundtrip(worker, target, expected, frame_count, run_dir):
    client = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    server = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    source = SampleVideo(target)
    client.addTrack(source)
    processed_track = None
    returned_track = asyncio.get_running_loop().create_future()

    @server.on('track')
    def on_server_track(track):
        nonlocal processed_track
        if track.kind != 'video' or processed_track is not None:
            raise RuntimeError('Expected exactly one video track.')
        processed_track = ProcessedVideoTrack(track, worker)
        server.addTrack(processed_track)

    @client.on('track')
    def on_client_track(track):
        if track.kind == 'video' and not returned_track.done():
            returned_track.set_result(track)

    try:
        await client.setLocalDescription(await client.createOffer())
        await server.setRemoteDescription(client.localDescription)
        await server.setLocalDescription(await server.createAnswer())
        await client.setRemoteDescription(server.localDescription)
        remote = await asyncio.wait_for(returned_track, 15)
        timestamps = []
        received = None
        for _ in range(frame_count):
            frame = await asyncio.wait_for(remote.recv(), 30)
            timestamps.append(float(frame.pts * frame.time_base))
            received = frame.to_ndarray(format='bgr24')
            if received.shape != target.shape:
                raise RuntimeError(f'Returned frame size changed: {received.shape}')
        if not all(b > a for a, b in zip(timestamps, timestamps[1:])):
            raise RuntimeError('Returned video timestamps did not advance.')
        original_error = float(np.abs(received.astype(float) - target).mean())
        processed_error = float(np.abs(received.astype(float) - expected).mean())
        if processed_error + 1 >= original_error:
            raise RuntimeError(f'Returned image does not match processing: {processed_error=} {original_error=}')
        client_stats, server_stats = await asyncio.gather(client.getStats(), server.getStats())
        received_packets = sum(s.packetsReceived for s in client_stats.values() if s.type == 'inbound-rtp')
        sent_packets = sum(s.packetsSent for s in server_stats.values() if s.type == 'outbound-rtp')
        if min(received_packets, sent_packets) <= 0:
            raise RuntimeError('No actual video RTP packets were transferred.')
        if client.connectionState != 'connected' or server.connectionState != 'connected':
            raise RuntimeError('WebRTC peers did not reach connected state.')
        if processed_track.error is not None:
            raise RuntimeError(f'Worker track failed: {processed_track.error}')
        if processed_track.dropped == 0:
            raise RuntimeError('Backpressure was not exercised; stale-frame dropping is unverified.')
        if not cv2.imwrite(str(run_dir / 'returned_frame.png'), received):
            raise RuntimeError('Could not save returned video frame.')
        return {'frames_received': frame_count, 'output_shape': list(received.shape),
                'client_state': client.connectionState, 'server_state': server.connectionState,
                'video_packets_received': received_packets, 'video_packets_sent': sent_packets,
                'frames_captured': processed_track.captured, 'frames_processed': processed_track.processed,
                'stale_frames_dropped': processed_track.dropped, 'pending_queue_limit': 1,
                'processing_p50_ms': float(np.percentile(processed_track.processing_ms, 50)),
                'processing_p95_ms': float(np.percentile(processed_track.processing_ms, 95)),
                'returned_vs_original_mae': original_error, 'returned_vs_processed_mae': processed_error,
                'timestamps_monotonic': True}
    finally:
        await asyncio.gather(client.close(), server.close())
        if processed_track is not None:
            await processed_track.aclose()
        source.stop()


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport-only', action='store_true')
    parser.add_argument('--source-image')
    parser.add_argument('--target-image')
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--output-dir', default='artifacts/webrtc-roundtrip')
    args = parser.parse_args()
    if not 3 <= args.frames <= 1000:
        parser.error('--frames must be between 3 and 1000')
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix='run-', dir=output_dir))
    if args.transport_only:
        worker = TransportTestWorker()
        target = np.full((128, 128, 3), 40, np.uint8)
    else:
        if not args.source_image or not args.target_image:
            parser.error('GPU mode requires --source-image and --target-image')
        from headless_cloud_worker import HeadlessCloudWorker
        worker = HeadlessCloudWorker(args.source_image, run_dir, require_cuda=True)
        target = cv2.imread(args.target_image)
        if target is None:
            raise ValueError('Cannot read target image.')
    try:
        expected, _ = await asyncio.to_thread(worker.process_frame, target)
        report = await roundtrip(worker, target, expected, args.frames, run_dir)
    finally:
        if not args.transport_only:
            await asyncio.to_thread(worker.close)
    report.update({'status': 'passed', 'scope': 'same-host WebRTC video round-trip',
                   'gpu_verified': not args.transport_only,
                   'external_windows_connection_tested': False})
    if not args.transport_only:
        report['execution_evidence'] = str(worker.profile_dir / 'execution_evidence.json')
    (run_dir / 'webrtc-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    print(f'WEBRTC ROUNDTRIP PASSED: {run_dir}', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
