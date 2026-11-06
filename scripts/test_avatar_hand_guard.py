import unittest
import numpy as np
from avatar_hand_guard import HandObservationGuard
from avatar_body_retarget import rotation_matrix
from test_avatar_head_hands import rest_hand


class HandGuardTests(unittest.TestCase):
    def test_global_hand_rotation_does_not_count_as_finger_jump(self):
        guard=HandObservationGuard();points=rest_hand('left')
        self.assertTrue(guard.check(points,0.)[0])
        self.assertTrue(guard.check(points @ rotation_matrix([1.,.6,.8]).T,33.)[0])

    def test_reversed_finger_rejected_without_poisoning_next_observation(self):
        guard=HandObservationGuard();points=rest_hand('left')
        guard.check(points,0.)
        bad=points.copy()
        for a,b in ((5,6),(6,7),(7,8)):bad[b]=bad[a]-(points[b]-points[a])
        self.assertFalse(guard.check(bad,33.)[0])
        self.assertTrue(guard.check(points,66.)[0])
        self.assertEqual(guard.timestamp,66.)


if __name__=='__main__':unittest.main()
