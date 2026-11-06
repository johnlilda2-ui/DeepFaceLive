"""Single-avatar, synchronous headless worker. No network or webcam tracking."""
import copy
import math
import threading

import numpy as np

SHAPES = {'root_pose': (3,), 'body_pose': (21, 3), 'jaw_pose': (3,),
          'leye_pose': (3,), 'reye_pose': (3,), 'lhand_pose': (15, 3),
          'rhand_pose': (15, 3), 'trans': (3,), 'focal': (2,),
          'princpt': (2,), 'img_size_wh': (2,), 'expr': (100,)}
OPTIONAL = {'expr', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose'}


def validate_pose(pose):
    if not isinstance(pose, dict) or set(pose) - set(SHAPES) - {'betas', 'pad_ratio'}:
        raise ValueError('Unsupported motion packet fields')
    result = {}
    for key, shape in SHAPES.items():
        if key not in pose:
            if key not in OPTIONAL:
                raise ValueError('Missing motion field: ' + key)
            value = np.zeros(shape, dtype=np.float32)
        else:
            value = np.asarray(pose[key])
            if value.shape != shape or value.dtype.kind not in 'fiu':
                raise ValueError('Invalid numeric shape: ' + key)
            if not np.isfinite(value).all() or (value.min() < -100000 or value.max() > 100000):
                raise ValueError('Nonfinite or excessive motion value: ' + key)
            value = value.astype(np.float32)
        result[key] = value.tolist()
    w, h = result['img_size_wh']
    if w != int(w) or h != int(h) or not 14 <= min(w, h) or max(w, h) > 8192 or max(w / h, h / w) > 4:
        raise ValueError('Unsupported camera image size')
    if min(result['focal']) <= 0:
        raise ValueError('Focal lengths must be positive')
    cx, cy = result['princpt']
    if not 0 <= cx <= w or not 0 <= cy <= h:
        raise ValueError('Principal point outside camera image')
    # Incoming shape coefficients are deliberately never forwarded.
    return result


class HeadlessAvatarWorker:
    """One immutable avatar per backend; reject concurrent work, never queue it."""
    def __init__(self, backend, initial_pose):
        if getattr(backend, '_dfl_worker_owner', None) is not None:
            raise ValueError('Backend already belongs to an avatar worker')
        backend._dfl_worker_owner = object()
        self._backend = backend
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._closed = False
        self._failed = False
        self._frames = 0
        self._last_id = -1
        self._last_timestamp = -1.0
        self._close_error = None
        try:
            initial = validate_pose(initial_pose)
            self._camera_size = initial['img_size_wh']
            backend.prepare(initial)
        except BaseException:
            self._failed = True
            self.close()
            raise

    def render(self, frame_id, timestamp_ms, pose):
        if self._closing.is_set() or self._closed:
            raise RuntimeError('Avatar worker is closed')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Avatar worker is busy; submit the latest frame later')
        try:
            if self._closing.is_set() or self._closed:
                raise RuntimeError('Avatar worker is closed')
            if (type(frame_id) is not int or not 0 <= frame_id < 2**53
                    or type(timestamp_ms) not in (int, float)
                    or not 0 <= timestamp_ms < 2**53 or not math.isfinite(timestamp_ms)):
                raise ValueError('Invalid frame ID or timestamp')
            if frame_id <= self._last_id or timestamp_ms <= self._last_timestamp:
                raise ValueError('Stale or duplicate motion frame')
            packet = validate_pose(pose)
            if packet['img_size_wh'] != self._camera_size:
                raise ValueError('Camera size changed; create a new worker')
            try:
                frame = self._backend.render(packet)
                if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                        or frame.shape != self._backend.output_shape):
                    raise RuntimeError('Backend returned an invalid RGB frame')
                frame = frame.copy()
            except BaseException:
                # Rendering failure invalidates cached state; never pass through
                # the source picture or continue with a potentially corrupt cache.
                self._failed = True
                self._closing.set()
                self._close_locked()
                raise
            self._last_id, self._last_timestamp = frame_id, timestamp_ms
            self._frames += 1
            return frame
        finally:
            self._lock.release()

    def _close_locked(self):
        if not self._closed:
            self._closed = True
            try:
                self._backend.close()
            except Exception as error:
                self._close_error = repr(error)

    def close(self):
        self._closing.set()
        with self._lock:
            self._close_locked()

    def snapshot(self):
        with self._lock:
            return {'frames_rendered': self._frames, 'last_frame_id': self._last_id,
                    'closed': self._closed, 'failed': self._failed,
                    'close_error': self._close_error,
                    'backend': copy.deepcopy(self._backend.stats)}
