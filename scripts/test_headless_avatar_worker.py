import threading
import unittest
import numpy as np
from headless_avatar_worker import HeadlessAvatarWorker, SHAPES, validate_pose


def packet():
    p = {k: np.zeros(s).tolist() for k, s in SHAPES.items()}
    p.update(focal=[1000, 1000], princpt=[768, 432], img_size_wh=[1536, 864])
    return p


class Backend:
    output_shape = (8, 8, 3)
    def __init__(self, color=100):
        self.stats = {'prepared': 0, 'renders': 0, 'closes': 0, 'cache_present': False}
        self.frame = np.full(self.output_shape, color, np.uint8)
        self.fail = False
        self.entered = self.release = None
    def prepare(self, pose):
        self.initial = pose
        self.stats['prepared'] += 1
        self.stats['cache_present'] = True
    def render(self, pose):
        if self.entered is not None:
            self.entered.set()
            if not self.release.wait(3):
                raise RuntimeError('Test render timeout')
        if self.fail:
            raise RuntimeError('GPU failure')
        self.latest = pose
        self.stats['renders'] += 1
        return self.frame
    def close(self):
        self.stats['closes'] += 1
        self.stats['cache_present'] = False


class WorkerTests(unittest.TestCase):
    def test_avatar_prepared_once_and_packets_cannot_change_identity(self):
        b = Backend()
        p = packet()
        w = HeadlessAvatarWorker(b, p)
        p['body_pose'][0][0] = 99
        self.assertEqual(b.initial['body_pose'][0][0], 0)
        p['betas'] = [999] * 10
        for i in range(18):
            w.render(i, i * 100, p)
        self.assertEqual(b.stats['prepared'], 1)
        self.assertEqual(b.stats['renders'], 18)
        self.assertNotIn('betas', b.latest)
        w.close()

    def test_stale_and_bad_frames_rejected_without_consuming_next_id(self):
        b = Backend(); w = HeadlessAvatarWorker(b, packet())
        w.render(0, 0, packet())
        for fid, ts in [(0, 1), (1, 0), (True, 2), (1, float('nan')), (1, 10**1000)]:
            with self.assertRaises(ValueError):
                w.render(fid, ts, packet())
        bad = packet(); bad['body_pose'][0][0] = float('inf')
        with self.assertRaises(ValueError):
            w.render(1, 1, bad)
        bad = packet(); bad['img_size_wh'] = [1280, 720]
        with self.assertRaises(ValueError):
            w.render(1, 1, bad)
        w.render(1, 1, packet())
        self.assertEqual(b.stats['renders'], 2)
        w.close()

    def test_frames_independent_and_backend_failure_closes_worker(self):
        b = Backend(); w = HeadlessAvatarWorker(b, packet())
        w.render(0, 0, packet())[:] = 0
        self.assertTrue((b.frame == 100).all())
        b.fail = True
        with self.assertRaisesRegex(RuntimeError, 'GPU failure'):
            w.render(1, 1, packet())
        with self.assertRaises(RuntimeError):
            w.render(2, 2, packet())
        w.close()
        self.assertEqual(b.stats['closes'], 1)
        self.assertTrue(w.snapshot()['failed'])

    def test_backend_ownership_and_separate_avatar_state(self):
        b1, b2 = Backend(50), Backend(150)
        w1, w2 = HeadlessAvatarWorker(b1, packet()), HeadlessAvatarWorker(b2, packet())
        with self.assertRaises(ValueError):
            HeadlessAvatarWorker(b1, packet())
        self.assertTrue((w1.render(0, 0, packet()) == 50).all())
        self.assertTrue((w2.render(0, 0, packet()) == 150).all())
        w1.close(); w2.close()

    def test_prepare_failure_and_invalid_backend_output_release_state(self):
        b = Backend()
        def fail(_): raise RuntimeError('prepare failed')
        b.prepare = fail
        with self.assertRaisesRegex(RuntimeError, 'prepare failed'):
            HeadlessAvatarWorker(b, packet())
        self.assertEqual(b.stats['closes'], 1)
        b = Backend(); w = HeadlessAvatarWorker(b, packet())
        b.frame = np.zeros((1, 1), np.uint8)
        with self.assertRaises(RuntimeError):
            w.render(0, 0, packet())
        self.assertEqual(b.stats['closes'], 1)

    def test_concurrent_render_rejected_and_close_waits_for_inflight_frame(self):
        b = Backend(); w = HeadlessAvatarWorker(b, packet())
        b.entered, b.release = threading.Event(), threading.Event()
        errors = []
        def render():
            try: w.render(0, 0, packet())
            except BaseException as e: errors.append(e)
        first = threading.Thread(target=render); first.start()
        self.assertTrue(b.entered.wait(2))
        with self.assertRaisesRegex(RuntimeError, 'busy'):
            w.render(1, 1, packet())
        closing = threading.Thread(target=w.close); closing.start()
        self.assertTrue(w._closing.wait(2))
        self.assertEqual(b.stats['closes'], 0)
        b.release.set()
        first.join(3); closing.join(3)
        self.assertFalse(first.is_alive() or closing.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(b.stats['closes'], 1)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            w.render(1, 1, packet())

    def test_invalid_camera_and_motion_shapes_rejected(self):
        for key, value in [('focal', [-1, 1]), ('body_pose', [0] * 63),
                           ('img_size_wh', [100000, 1]), ('princpt', [9999, 1]), ('trans', [-(2**63), 0, 0])]:
            p = packet(); p[key] = value
            with self.assertRaises(ValueError):
                validate_pose(p)


if __name__ == '__main__':
    unittest.main()
