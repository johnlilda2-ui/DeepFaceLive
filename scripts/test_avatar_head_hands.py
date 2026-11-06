import copy
import unittest
import numpy as np
from avatar_head_hands import NeutralHeadCalibration, HandRetargeter
from avatar_body_retarget import rotation_matrix
from test_avatar_body_retarget import template


def rest_hand(side):
    sign = 1 if side == 'left' else -1
    points = np.zeros((21,3))
    for ids,x,z in [((5,6,7,8),.050,.025),((9,10,11,12),.055,0),
                    ((13,14,15,16),.050,-.020),((17,18,19,20),.045,-.038)]:
        for i,idx in enumerate(ids):
            points[idx] = [sign*(x+[0,.025,.045,.060][i]),0,z]
    for i,idx in enumerate((1,2,3,4)):
        points[idx] = [sign*(.020+i*.017),0,.020+i*.014]
    return points


class HeadHandTests(unittest.TestCase):
    def test_neutral_calibration_removes_feature_frame_pitch_bias(self):
        p = template()
        p['body_pose'][14] = [.7,0,0]
        calibration = NeutralHeadCalibration()
        calibration.calibrate(p)
        fixed = calibration.apply(p)
        np.testing.assert_allclose(fixed['body_pose'][14],0,atol=1e-7)
        self.assertEqual(p['body_pose'][14],[.7,0,0])

    def test_calibration_preserves_composed_relative_nod_and_turn(self):
        from avatar_body_retarget import _matrix_to_axis_angle
        p = template(); bias = rotation_matrix([.7,.1,0])
        p['body_pose'][14] = _matrix_to_axis_angle(bias).tolist()
        c = NeutralHeadCalibration(); c.calibrate(p)
        delta = rotation_matrix([.2,-.3,.05])
        p['body_pose'][14] = _matrix_to_axis_angle(delta @ bias).tolist()
        fixed = c.apply(p)
        np.testing.assert_allclose(rotation_matrix(fixed['body_pose'][14]),delta,atol=1e-7)
        for i in range(21):
            if i != 14: self.assertEqual(fixed['body_pose'][i],p['body_pose'][i])

    def test_uncalibrated_head_is_rejected(self):
        with self.assertRaises(ValueError):
            NeutralHeadCalibration().apply(template())

    def test_rest_hands_have_identity_finger_and_wrist_rotations(self):
        for side in ('left','right'):
            p=template(); p['root_pose']=[np.pi,0,0]
            result=HandRetargeter(rest_hand(side),side,1).apply(p,rest_hand(side))
            np.testing.assert_allclose(result['body_pose'],0,atol=1e-7)
            np.testing.assert_allclose(result['lhand_pose'],0,atol=1e-7)
            np.testing.assert_allclose(result['rhand_pose'],0,atol=1e-7)

    def test_global_palm_rotation_does_not_bend_fingers(self):
        for side in ('left','right'):
            p=template(); p['root_pose']=[np.pi,0,0]
            rest=rest_hand(side); r=rotation_matrix([.2,-.4,.3])
            result=HandRetargeter(rest,side,1).apply(p,rest @ r.T)
            idx=19 if side=='left' else 20
            np.testing.assert_allclose(rotation_matrix(result['body_pose'][idx]),r,atol=1e-7)
            np.testing.assert_allclose(result['lhand_pose' if side=='left' else 'rhand_pose'],0,atol=1e-7)

    def test_finger_curl_uses_local_parent_rotations(self):
        for side in ('left','right'):
            p=template(); p['root_pose']=[np.pi,0,0]
            rest=rest_hand(side); moved=rest.copy()
            for a,b,angle in ((5,6,.5),(6,7,.9),(7,8,1.2)):
                moved[b]=moved[a]+rotation_matrix([0,0,angle]) @ (rest[b]-rest[a])
            result=HandRetargeter(rest,side,1).apply(p,moved)
            hand=result['lhand_pose' if side=='left' else 'rhand_pose']
            for i,angle in enumerate((.5,.4,.3)):
                np.testing.assert_allclose(rotation_matrix(hand[i]),rotation_matrix([0,0,angle]),atol=1e-7)
            np.testing.assert_allclose(hand[3:],0,atol=1e-7)

    def test_bad_hand_does_not_poison_previous_state(self):
        p=template(); p['root_pose']=[np.pi,0,0]; rest=rest_hand('left')
        retarget=HandRetargeter(rest,'left')
        retarget.apply(p,rest)
        old=copy.deepcopy(retarget.previous)
        bad=rest.copy(); bad[6]=bad[5]
        with self.assertRaises(ValueError): retarget.apply(p,bad)
        for a,b in zip(old,retarget.previous): np.testing.assert_array_equal(a,b)
        retarget.reset(); self.assertIsNone(retarget.previous)


if __name__ == '__main__':
    unittest.main()
