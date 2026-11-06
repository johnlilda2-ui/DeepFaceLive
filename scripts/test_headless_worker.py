"""Frame-contract and compositing regressions without ONNX or a GPU."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import headless_cloud_worker as worker_module


class IdentityTransform:
    def invert(self):
        return self

    def to_exact_mat(self, *sizes):
        return np.float32([[1, 0, 0], [0, 1, 0]])


class WorkerTests(unittest.TestCase):
    def make_worker(self):
        worker = worker_module.HeadlessCloudWorker.__new__(worker_module.HeadlessCloudWorker)
        worker.closed = False
        worker.frame_count = 0
        worker.threshold = .5
        worker.detector = worker.marker = worker.swap_model = None
        return worker

    def test_premultiplied_blend_preserves_constant_colors_at_edges(self):
        frame = np.full((128, 128, 3), 100, np.uint8)
        result = worker_module.composite_face(frame, frame, IdentityTransform())
        np.testing.assert_array_equal(result, frame)

    def test_blend_preserves_background_and_changes_face_center(self):
        frame = np.zeros((128, 128, 3), np.uint8)
        swapped = np.full_like(frame, 200)
        result = worker_module.composite_face(frame, swapped, IdentityTransform())
        np.testing.assert_array_equal(result[0, 0], frame[0, 0])
        self.assertGreater(result[64, 64].min(), 190)
        self.assertEqual(result.shape, frame.shape)
        self.assertEqual(result.dtype, frame.dtype)

    def test_mouth_preservation_keeps_original_lips_and_swaps_surrounding_face(self):
        frame = np.full((128,128,3), 30, np.uint8)
        swapped = np.full_like(frame,220)
        angles = np.linspace(0,2*np.pi,20,endpoint=False)
        mouth = np.column_stack((64+15*np.cos(angles),80+7*np.sin(angles)))
        result = worker_module.composite_face(frame,swapped,IdentityTransform(),mouth)
        np.testing.assert_array_equal(result[80,64],frame[80,64])
        self.assertGreater(result[50,64].min(),200)

    def test_invalid_identity_does_not_replace_existing_vector(self):
        worker = self.make_worker()
        worker.face_vector = 'previous'
        worker.detector = SimpleNamespace(extract=lambda *a,**kw: [[]])
        with self.assertRaisesRegex(ValueError,'exactly one'):
            worker.set_identity(np.zeros((128,128,3),np.uint8))
        self.assertEqual(worker.face_vector,'previous')

    def test_no_face_is_an_independent_passthrough_copy(self):
        worker = self.make_worker()
        frame = np.zeros((64, 64, 3), np.uint8)
        with patch.object(worker_module.smoke, 'detect_and_align_face',
                          side_effect=worker_module.smoke.NoFaceDetected()):
            result, status = worker.process_frame(frame)
        self.assertEqual(status, 'no_face')
        np.testing.assert_array_equal(result, frame)
        self.assertFalse(np.shares_memory(result, frame))

    def test_inference_errors_are_not_treated_as_no_face(self):
        worker = self.make_worker()
        with patch.object(worker_module.smoke, 'detect_and_align_face',
                          side_effect=RuntimeError('GPU execution failed')):
            with self.assertRaisesRegex(RuntimeError, 'GPU execution failed'):
                worker.process_frame(np.zeros((64, 64, 3), np.uint8))

    def test_invalid_frames_and_closed_worker_are_rejected(self):
        worker = self.make_worker()
        for frame in [None, np.zeros((64, 64), np.uint8), np.zeros((64, 64, 3), np.float32)]:
            with self.assertRaises(ValueError):
                worker.process_frame(frame)
        worker.closed = True
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            worker.process_frame(np.zeros((64, 64, 3), np.uint8))


if __name__ == '__main__':
    unittest.main()
