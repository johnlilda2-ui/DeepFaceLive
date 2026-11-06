"""Backpressure, timestamp, error, and shutdown checks without network or GPU."""
import asyncio
from fractions import Fraction
import threading
import time
import unittest

import numpy as np
from aiortc import MediaStreamTrack
from av import VideoFrame

from headless_webrtc_track import ProcessedVideoTrack


class Source(MediaStreamTrack):
    kind = 'video'
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.five_frames = asyncio.Event()

    async def recv(self):
        await asyncio.sleep(.005)
        frame = VideoFrame.from_ndarray(np.full((64, 64, 3), 40, np.uint8), format='bgr24')
        frame.pts, frame.time_base = self.counter * 3000, Fraction(1, 90000)
        self.counter += 1
        if self.counter >= 5:
            self.five_frames.set()
        return frame


class TrackTests(unittest.IsolatedAsyncioTestCase):
    async def test_bypass_returns_original_without_inference(self):
        class UnusedWorker:
            def process_frame(self, image):
                raise AssertionError('Bypass called inference')
        track = ProcessedVideoTrack(Source(), UnusedWorker(), swap_enabled=False)
        try:
            output = await track.recv()
            self.assertEqual(int(output.to_ndarray(format='bgr24')[0,0,0]),40)
            self.assertEqual(track.last_status,'bypass')
            self.assertEqual(track.swapped_frames,0)
        finally:
            await track.aclose()
    async def test_backpressure_and_timestamps(self):
        release = threading.Event()
        class GatedWorker:
            def process_frame(self, frame):
                if not release.wait(2):
                    raise RuntimeError('Test did not release inference')
                return 255 - frame, 'processed'
        source = Source()
        track = ProcessedVideoTrack(source, GatedWorker())
        receiving = asyncio.create_task(track.recv())
        try:
            # Force five incoming frames while first inference is blocked. This
            # validates backpressure independently of CI scheduling or CPU speed.
            await asyncio.wait_for(source.five_frames.wait(), 2)
            self.assertGreater(track.dropped, 0)
            self.assertLessEqual(track.pending.qsize(), 1)
            release.set()
            first = await asyncio.wait_for(receiving, 2)
            second = await asyncio.wait_for(track.recv(), 2)
            self.assertGreaterEqual(second.pts, first.pts + 12000)
            self.assertEqual(second.time_base, Fraction(1, 90000))
            self.assertEqual(int(second.to_ndarray(format='bgr24')[0, 0, 0]), 215)
        finally:
            release.set()
            await asyncio.gather(receiving, return_exceptions=True)
            await track.aclose()
        self.assertEqual(source.readyState, 'ended')

    async def test_processing_failure_propagates(self):
        class Broken:
            def process_frame(self, frame):
                raise RuntimeError('inference failed')
        track = ProcessedVideoTrack(Source(), Broken())
        try:
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                await asyncio.wait_for(track.recv(), 2)
            self.assertEqual(track.readyState, 'ended')
            self.assertIsInstance(track.error, RuntimeError)
        finally:
            await track.aclose()

    async def test_shutdown_waits_for_cancelled_inference(self):
        started, finished = threading.Event(), threading.Event()
        class InFlight:
            def process_frame(self, frame):
                started.set()
                time.sleep(.1)
                finished.set()
                return frame, 'processed'
        track = ProcessedVideoTrack(Source(), InFlight())
        receiving = asyncio.create_task(track.recv())
        self.assertTrue(await asyncio.to_thread(started.wait, 2))
        receiving.cancel()
        await asyncio.gather(receiving, return_exceptions=True)
        await track.aclose()
        self.assertTrue(finished.is_set())
        self.assertIsNone(track.processing)
        self.assertTrue(track.reader.done())

    async def test_shutdown_does_not_hide_pending_inference_failure(self):
        started = threading.Event()
        class InFlightFailure:
            def process_frame(self, frame):
                started.set()
                time.sleep(.08)
                raise RuntimeError('late GPU failure')
        track = ProcessedVideoTrack(Source(), InFlightFailure())
        receiving = asyncio.create_task(track.recv())
        self.assertTrue(await asyncio.to_thread(started.wait, 2))
        receiving.cancel()
        await asyncio.gather(receiving, return_exceptions=True)
        with self.assertRaisesRegex(RuntimeError, 'late GPU failure'):
            await track.aclose()


if __name__ == '__main__':
    unittest.main()
