"""Retargeting geometry tests; neutral fixture adapted from Apache-2.0 PoseCap."""
import unittest
import numpy as np
from avatar_body_retarget import (LandmarkPoseConverter, MediaPipeBodyRetargeter,
    LANDMARK_INDICES, CAMERA_ALIGNMENT, rotation_matrix)
from headless_avatar_worker import SHAPES

def _neutral_landmarks() -> dict[str, tuple[float, float, float]]:
    return {
        "left_hip": (0.2, 0.0, 0.0),
        "right_hip": (-0.2, 0.0, 0.0),
        "left_shoulder": (0.4, 1.0, 0.0),
        "right_shoulder": (-0.4, 1.0, 0.0),
        "left_elbow": (0.7, 1.0, 0.0),
        "right_elbow": (-0.7, 1.0, 0.0),
        "left_wrist": (1.0, 1.0, 0.0),
        "right_wrist": (-1.0, 1.0, 0.0),
        "left_knee": (0.2, -1.0, 0.0),
        "right_knee": (-0.2, -1.0, 0.0),
        "left_ankle": (0.2, -2.0, 0.0),
        "right_ankle": (-0.2, -2.0, 0.0),
        "left_heel": (0.2, -2.0, -0.1),
        "right_heel": (-0.2, -2.0, -0.1),
        "left_foot_index": (0.2, -2.0, 0.3),
        "right_foot_index": (-0.2, -2.0, 0.3),
        "left_ear": (0.1, 1.55, 0.0),
        "right_ear": (-0.1, 1.55, 0.0),
        "mouth_left": (0.04, 1.4, 0.0),
        "mouth_right": (-0.04, 1.4, 0.0),
        "nose": (0.0, 1.5, 0.0),
    }



def template():
    p = {k: np.zeros(s).tolist() for k, s in SHAPES.items()}
    p.update(focal=[1330, 1330], princpt=[768, 432], img_size_wh=[1536, 864],
             trans=[0, 0.3, 5])
    return p


def world(points):
    out = np.zeros((33, 3))
    for name, idx in LANDMARK_INDICES.items():
        out[idx] = np.array(points[name]) * [1, -1, -1]
    return out


class BodyRetargetTests(unittest.TestCase):
    def test_neutral_body_and_camera_rotation(self):
        p = MediaPipeBodyRetargeter(template()).convert(world(_neutral_landmarks()), np.ones(33))
        np.testing.assert_allclose(p['body_pose'], 0, atol=1e-7)
        np.testing.assert_allclose(rotation_matrix(p['root_pose']), CAMERA_ALIGNMENT, atol=1e-7)
        self.assertNotIn('betas', p)
        self.assertTrue(np.all(np.asarray(p['expr']) == 0))

    def test_global_rotation_composes_instead_of_adding_axis_angles(self):
        r = rotation_matrix([0, 0.6, 0])
        points = {k: r @ v for k, v in _neutral_landmarks().items()}
        p = MediaPipeBodyRetargeter(template()).convert(world(points), np.ones(33))
        np.testing.assert_allclose(rotation_matrix(p['root_pose']), CAMERA_ALIGNMENT @ r, atol=1e-7)
        np.testing.assert_allclose(p['body_pose'], 0, atol=1e-7)

    def test_left_arm_vertical_pole_and_continuity(self):
        converter = LandmarkPoseConverter()
        for degrees in (0, 60, 75, 90, 105, 120, 150, 180):
            points = _neutral_landmarks()
            r = rotation_matrix([0, 0, np.radians(degrees)])
            shoulder = np.array(points['left_shoulder'])
            points['left_elbow'] = shoulder + r @ [0.3, 0, 0]
            points['left_wrist'] = shoulder + r @ [0.6, 0, 0]
            p = converter.convert(points)
            np.testing.assert_allclose(rotation_matrix(p['body_pose'][15]), r, atol=1e-7)
            np.testing.assert_allclose(p['body_pose'][16], 0, atol=1e-7)
            np.testing.assert_allclose(p['body_pose'][17], 0, atol=1e-7)

    def test_right_arm_down_does_not_get_inverted_twist(self):
        points = _neutral_landmarks()
        points['right_elbow'] = [-0.4, 0.7, 0]
        points['right_wrist'] = [-0.4, 0.4, 0]
        p = LandmarkPoseConverter().convert(points)
        np.testing.assert_allclose(rotation_matrix(p['body_pose'][16]),
                                   rotation_matrix([0, 0, np.pi / 2]), atol=1e-7)

    def test_sideways_leg_keeps_actual_bone_direction(self):
        points = _neutral_landmarks()
        points.update(left_knee=[1.2, 0, 0], left_ankle=[2.2, 0, 0],
                      left_heel=[2.2, -0.1, -0.1], left_foot_index=[2.2, -0.1, 0.3])
        p = LandmarkPoseConverter().convert(points)
        hip = rotation_matrix(p['body_pose'][0])
        np.testing.assert_allclose(hip @ [0, -1, 0], [1, 0, 0], atol=1e-7)
        np.testing.assert_allclose(p['body_pose'][3], 0, atol=1e-7)

    def test_confidence_loss_and_nonfinite_landmarks_do_not_make_a_packet(self):
        tracker = MediaPipeBodyRetargeter(template())
        pts = world(_neutral_landmarks())
        confidence = np.ones(33); confidence[15] = 0.49
        self.assertIsNone(tracker.convert(pts, confidence))
        pts[0, 0] = np.nan
        self.assertIsNone(tracker.convert(pts, np.ones(33)))

    def test_camera_is_copied_and_untracked_channels_remain_neutral(self):
        camera = template()
        tracker = MediaPipeBodyRetargeter(camera)
        camera['trans'][0] = 99
        p = tracker.convert(world(_neutral_landmarks()), np.ones(33))
        self.assertEqual(p['trans'], [0, np.float32(0.3), 5])
        np.testing.assert_allclose(p['lhand_pose'], 0)
        np.testing.assert_allclose(p['rhand_pose'], 0)

    def test_degenerate_input_is_rejected(self):
        with self.assertRaises(ValueError):
            LandmarkPoseConverter().convert({k: [0, 0, 0] for k in _neutral_landmarks()})


if __name__ == '__main__':
    unittest.main()
