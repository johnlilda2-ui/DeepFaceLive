import unittest
import numpy as np
from lhmpp_precision_quality import compare_frames

class PrecisionQualityTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.full((2, 20, 20, 3), 255, np.uint8)
        self.reference[:, 5:15, 5:15] = 100

    def test_identical_sequence_passes(self):
        r = compare_frames(self.reference, self.reference.copy())
        self.assertTrue(r['quality_gate_passed'])
        self.assertEqual(r['full_mae'], 0)

    def test_subject_damage_is_not_hidden_by_white_background(self):
        candidate = self.reference.copy()
        candidate[:, 5:15, 5:15] += 3
        r = compare_frames(self.reference, candidate)
        self.assertLess(r['full_mae'], 1)
        self.assertFalse(r['quality_gate_passed'])
        self.assertEqual(r['subject_mae'], 3)

    def test_missing_subject_and_outline_change_rejected(self):
        with self.assertRaises(ValueError):
            compare_frames(np.full_like(self.reference, 255), self.reference)
        candidate = self.reference.copy()
        candidate[:, 5:15, 15:16] = 100
        self.assertFalse(compare_frames(self.reference, candidate)['quality_gate_passed'])

if __name__ == '__main__':
    unittest.main()
