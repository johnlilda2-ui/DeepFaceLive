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
