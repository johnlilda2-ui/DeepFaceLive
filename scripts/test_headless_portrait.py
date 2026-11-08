"""CPU contract checks; these do not claim portrait model inference."""
import unittest
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np

import headless_portrait_worker as portrait
from headless_lia_portrait_worker import HeadlessLiaPortraitWorker


class IdentityTransform:
    def invert(self):
        return self
    def to_exact_mat(self,*sizes):
        return np.float32([[1,0,0],[0,1,0]])


class PortraitTests(unittest.TestCase):
    def test_portrait_notebook_code_compiles(self):
        notebook = json.loads((Path(__file__).resolve().parents[1]/'notebooks/kaggle_portrait_avatar_test.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']),cell['id'],'exec')

    def test_portrait_paste_changes_head_crop_and_preserves_source_outside(self):
        source = np.full((256,256,3),40,np.uint8)
        generated = np.full((128,128,3),220,np.uint8)
        output = portrait.paste_portrait(source,generated,IdentityTransform())
        np.testing.assert_array_equal(output[200,200],source[200,200])
        self.assertGreater(output[30,30].min(),200)
        self.assertEqual(output.shape,source.shape)

    def test_head_blend_aligns_neck_and_cleans_only_the_tracked_person(self):
        from head_replacement import composite_head

        class Segmenter:
            def regions(self, image):
                return tuple((image[..., 0] == value).astype(np.float32)
                             for value in (200, 130, 220, 180))

        frame = np.full((256, 256, 3), 40, np.uint8)
        frame[20:80, 20:80] = 220
        frame[30:43, 80:82] = 180
        frame[80:106, 32:68] = 130
        frame[20:80, 170:230] = 220
        generated = np.zeros((128, 128, 3), np.uint8)
        generated[25:85, 30:90] = [220, 90, 100]
        generated[85:100, 48:72] = [130, 190, 210]
        output = composite_head(frame, frame[:128, :128], generated, IdentityTransform(), Segmenter())
        np.testing.assert_array_equal(output[21, 21], [220, 90, 100])
        self.assertGreater(int(output[82, 50, 2]), 180)
        self.assertGreater(int(output[92, 35, 2]), int(output[98, 35, 2]))
        self.assertGreater(int(output[98, 35, 2]), int(frame[98, 35, 2]))
        np.testing.assert_array_equal(output[101:, :], frame[101:, :])
        self.assertLess(int(output[36, 81, 0]), 100)
        np.testing.assert_array_equal(output[:, 84:], frame[:, 84:])
        self.assertIsNone(composite_head(frame, frame[:128, :128], np.zeros_like(generated),
                                         IdentityTransform(), Segmenter()))

    def test_missing_driver_face_holds_portrait_instead_of_exposing_driver(self):
        worker = portrait.HeadlessPortraitWorker.__new__(portrait.HeadlessPortraitWorker)
        worker.closed = False
        worker.source = np.full((128,128,3),120,np.uint8)
        worker.last_output = worker.source.copy()
        worker.detector = worker.marker = None
        worker.threshold = .5
        with patch.object(portrait.smoke,'detect_and_align_face',side_effect=portrait.smoke.NoFaceDetected()):
            output,status = worker.process_frame(np.zeros((64,64,3),np.uint8))
        self.assertEqual(status,'avatar_no_face')
        np.testing.assert_array_equal(output,worker.source)
        self.assertFalse(np.shares_memory(output,worker.last_output))
        worker.driving_initial = object()
        worker.reset_motion()
        self.assertIsNone(worker.driving_initial)

    def test_real_mouth_overlay_is_rejected_for_portrait(self):
        worker = portrait.HeadlessPortraitWorker.__new__(portrait.HeadlessPortraitWorker)
        worker.closed = False
        with self.assertRaisesRegex(ValueError,'face-swap only'):
            worker.process_frame(np.zeros((64,64,3),np.uint8),preserve_mouth=True)

    def test_lia_missing_face_and_reset_never_return_driver(self):
        worker = HeadlessLiaPortraitWorker.__new__(HeadlessLiaPortraitWorker)
        worker.closed = False
        worker.source = np.full((256,256,3),140,np.uint8)
        worker.last_output = worker.source.copy()
        worker.detector = worker.marker = None
        worker.threshold = .5
        with patch.object(portrait.smoke,'detect_and_align_face',side_effect=portrait.smoke.NoFaceDetected()):
            output,status = worker.process_frame(np.zeros((64,64,3),np.uint8))
        self.assertEqual(status,'avatar_no_face')
        np.testing.assert_array_equal(output,worker.source)
        worker.driving_initial = object()
        worker.reset_motion()
        self.assertIsNone(worker.driving_initial)


if __name__ == '__main__':
    unittest.main()
