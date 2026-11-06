import copy
import math
import unittest
from avatar_mouth_motion import JawOpening


class JawTests(unittest.TestCase):
    def test_preserves_identity_body_hands_and_expressions(self):
        packet = dict(body_pose=[[0, 0, 1]], lhand_pose=[[1, 2, 3]],
                      expr=[.2]*100, jaw_pose=[0, 0, 0])
        original = copy.deepcopy(packet)
        out = JawOpening(smoothing=1).apply(packet, 1.)
        self.assertEqual(packet, original)
        self.assertEqual(out['jaw_pose'], [.4, 0., 0.])
        for key in ('body_pose', 'lhand_pose', 'expr'):
            self.assertEqual(out[key], original[key])

    def test_missing_face_relaxes_and_invalid_input_cannot_poison_state(self):
        tracker = JawOpening(smoothing=.5)
        self.assertEqual(tracker.apply({}, 1.)['jaw_pose'], [.2, 0., 0.])
        for bad in (math.nan, math.inf, -1., 1.1, True, '1'):
            with self.assertRaises(ValueError):
                tracker.apply({}, bad)
            self.assertEqual(tracker.previous, .2)
        self.assertEqual(tracker.missing({})['jaw_pose'], [.1, 0., 0.])


if __name__ == '__main__':
    unittest.main()
