"""WebRTC video adapter for the existing serial cloud worker.

No network listener or signaling service is started on import. The adapter keeps
one pending frame, dropping stale video instead of accumulating delay.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque

from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame


class ProcessedVideoTrack(MediaStreamTrack):
    kind = 'video'

    def __init__(self, source, worker, swap_enabled=True, preserve_mouth=False):
        super().__init__()
        self.source = source
        self.worker = worker
        self.swap_enabled = swap_enabled
        self.preserve_mouth = preserve_mouth
        self.last_status = 'waiting'
        self.swapped_frames = self.no_face_frames = 0
        self.avatar_frames = 0
        self.pending = asyncio.Queue(maxsize=1)
        self.reader = None
        self.processing = None
        self.captured = self.processed = self.dropped = 0
        self.processing_ms = deque(maxlen=256)
        self.error = None
        self._inference_lock = asyncio.Lock()

    def _replace_pending(self, item):
        if self.pending.full():
            self.pending.get_nowait()
            self.dropped += 1
        self.pending.put_nowait(item)

    async def _read(self):
        try:
            while self.readyState == 'live':
                frame = await self.source.recv()
                self.captured += 1
                self._replace_pending(frame)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._replace_pending(exc)

    async def _process(self, frame):
        async with self._inference_lock:
            started = time.perf_counter()
            try:
                image = frame.to_ndarray(format='bgr24')
                if not self.swap_enabled:
                    result, status = image, 'bypass'
                elif hasattr(self.worker, 'process_video'):
                    if self.preserve_mouth:
                        raise ValueError('Source-mouth overlay is unavailable for this worker')
                    result, status = await asyncio.to_thread(self.worker.process_video, image,
                                                            float(frame.pts * frame.time_base) * 1000)
                elif self.preserve_mouth:
                    result, status = await asyncio.to_thread(self.worker.process_frame, image, preserve_mouth=True)
                else:
                    result, status = await asyncio.to_thread(self.worker.process_frame, image)
            except Exception as exc:
                self.error = exc
                raise
            self.processing_ms.append((time.perf_counter() - started) * 1000)
            self.processed += 1
            self.last_status = status
            self.swapped_frames += status == 'swapped'
            self.no_face_frames += status in ('no_face','avatar_no_face')
            self.avatar_frames += status == 'avatar'
            output = VideoFrame.from_ndarray(result, format='bgr24')
            output.pts, output.time_base = frame.pts, frame.time_base
            return output

    async def recv(self):
        if self.readyState != 'live':
            raise MediaStreamError
        if self.reader is None:
            self.reader = asyncio.create_task(self._read())
        item = await self.pending.get()
        if isinstance(item, Exception):
            self.error = item
            self.stop()
            raise item
        if item.pts is None or item.time_base is None:
            self.error = ValueError('Incoming video frame has no timestamp.')
            self.stop()
            raise self.error
        self.processing = asyncio.create_task(self._process(item))
        try:
            # Cancelling recv must not leave inference running after worker.close().
            return await asyncio.shield(self.processing)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = exc
            self.stop()
            raise
        finally:
            if self.processing.done():
                self.processing = None

    def stop(self):
        super().stop()
        self.source.stop()
        if self.reader is not None:
            self.reader.cancel()

    async def aclose(self):
        self.stop()
        if self.reader is not None:
            await asyncio.gather(self.reader, return_exceptions=True)
        if self.processing is not None:
            results = await asyncio.gather(self.processing, return_exceptions=True)
            self.processing = None
            if isinstance(results[0], Exception):
                raise results[0]

