"""Body-only landmark retargeting, adapted from Apache-2.0 PoseCap.

Upstream: CorridorTech/PoseCap @ 626701bd6315727e791cdc9c1353830b9492d6ee
Modified: standalone dictionary packets, stable limb poles, direction-preserving
leg frames, foot tripod frames, confidence gating and LHM++ camera alignment.
See third_party/posecap/LICENSE and NOTICE.md. Limb twist is approximate;
this does not implement finger motion, facial expressions or lip synchronization.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

import numpy as np
NUM_BETAS, NUM_EXPRESSION, NUM_HAND_JOINTS = 10, 100, 15
BODY_JOINT_NAMES = ["left_hip","right_hip","spine1","left_knee","right_knee","spine2","left_ankle","right_ankle","spine3","left_foot","right_foot","neck","left_collar","right_collar","head","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist"]

class PoseCapError(ValueError):
    pass


def quaternion_to_axis_angle(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise PoseCapError('Zero-norm quaternion')
    q = q / norm
    if q[0] < 0:
        q = -q
    sine = np.linalg.norm(q[1:])
    if sine < 1e-12:
        return np.zeros(3)
    return q[1:] * (2 * np.arctan2(sine, np.clip(q[0], -1, 1)) / sine)

LandmarkMap: TypeAlias = Mapping[str, Sequence[float]]

_EPSILON = 1e-8
_IDENTITY = np.eye(3, dtype=np.float64)

_PARENTS = {
    "left_hip": "pelvis",
    "right_hip": "pelvis",
    "spine1": "pelvis",
    "left_knee": "left_hip",
    "right_knee": "right_hip",
    "spine2": "spine1",
    "left_ankle": "left_knee",
    "right_ankle": "right_knee",
    "spine3": "spine2",
    "left_foot": "left_ankle",
    "right_foot": "right_ankle",
    "neck": "spine3",
    "left_collar": "spine3",
    "right_collar": "spine3",
    "head": "neck",
    "left_shoulder": "left_collar",
    "right_shoulder": "right_collar",
    "left_elbow": "left_shoulder",
    "right_elbow": "right_shoulder",
    "left_wrist": "left_elbow",
    "right_wrist": "right_elbow",
}


class LandmarkPoseConverter:
    """Stateful canonical-landmark to SMPL-X-order body rotation adapter.

    The input axes are PoseCap canonical axes: +X anatomical left, +Y up,
    +Z forward. Long-bone axial twist is resolved from the parent anatomical
    frame; it is an approximation because joint centres do not observe twist.
    """

    def __init__(self):
        self._previous_frames = {}

    def convert(self, landmarks: LandmarkMap) -> dict:
        """Convert one landmark map to a pelvis-locked SMPL-X pose payload (axis-angle, radians)."""
        points = {name: _point(landmarks, name) for name in _required_landmarks()}
        global_frames = _global_frames(points, self._previous_frames)
        self._previous_frames = {key: value.copy() for key, value in global_frames.items()}
        body_pose = [
            _matrix_to_axis_angle(global_frames[_PARENTS[name]].T @ global_frames[name]).tolist()
            for name in BODY_JOINT_NAMES
        ]
        zero_hand = [[0.0, 0.0, 0.0] for _ in range(NUM_HAND_JOINTS)]
        return dict(
            global_orient=_matrix_to_axis_angle(global_frames["pelvis"]).tolist(),
            body_pose=body_pose,
            left_hand_pose=[row.copy() for row in zero_hand],
            right_hand_pose=[row.copy() for row in zero_hand],
            jaw_pose=[0.0, 0.0, 0.0],
            betas=[0.0] * NUM_BETAS,
            expression=[0.0] * NUM_EXPRESSION,
            transl=[0.0, 0.0, 0.0],
        )


def _global_frames(points: dict[str, np.ndarray], previous) -> dict[str, np.ndarray]:
    hip_center = _midpoint(points["left_hip"], points["right_hip"])
    shoulder_center = _midpoint(points["left_shoulder"], points["right_shoulder"])
    pelvis = _frame_from_xy(points["left_hip"] - points["right_hip"], shoulder_center - hip_center)
    chest = _frame_from_xy(
        points["left_shoulder"] - points["right_shoulder"], shoulder_center - hip_center
    )
    head = _frame_from_xy(
        points["left_ear"] - points["right_ear"],
        points["nose"] - _midpoint(points["mouth_left"], points["mouth_right"]),
    )
    frames: dict[str, np.ndarray] = {
        "pelvis": pelvis,
        "spine1": pelvis,
        "spine2": pelvis,
        "spine3": chest,
        "neck": chest,
        "head": head,
        "left_collar": chest,
        "right_collar": chest,
    }
    for side in ("left", "right"):
        hip = points[f"{side}_hip"]
        knee = points[f"{side}_knee"]
        ankle = points[f"{side}_ankle"]
        foot_index = points[f"{side}_foot_index"]
        hip_frame = _frame_from_yx(hip - knee, pelvis[:, 0], previous.get(f'{side}_hip'), pelvis[:, 2])
        knee_frame = _frame_from_yx(knee - ankle, hip_frame[:, 0], previous.get(f'{side}_knee'), hip_frame[:, 2])
        ankle_frame = _foot_frame(ankle, points[f'{side}_heel'], foot_index)
        frames[f"{side}_hip"] = hip_frame
        frames[f"{side}_knee"] = knee_frame
        frames[f"{side}_ankle"] = ankle_frame
        frames[f"{side}_foot"] = ankle_frame

        shoulder = points[f"{side}_shoulder"]
        elbow = points[f"{side}_elbow"]
        wrist = points[f"{side}_wrist"]
        direction_sign = 1.0 if side == "left" else -1.0
        shoulder_frame = _frame_from_xy(direction_sign * (elbow - shoulder), chest[:, 1], previous.get(f'{side}_shoulder'), chest[:, 2])
        elbow_frame = _frame_from_xy(direction_sign * (wrist - elbow), shoulder_frame[:, 1], previous.get(f'{side}_elbow'), shoulder_frame[:, 2])
        frames[f"{side}_shoulder"] = shoulder_frame
        frames[f"{side}_elbow"] = elbow_frame
        frames[f"{side}_wrist"] = elbow_frame
    return frames


def _required_landmarks() -> tuple[str, ...]:
    return (
        "left_hip",
        "right_hip",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
        "nose",
    )


def _point(landmarks: LandmarkMap, name: str) -> np.ndarray:
    try:
        point = np.asarray(landmarks[name], dtype=np.float64)
    except KeyError as error:
        raise PoseCapError(f"missing landmark: {name}") from error
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise PoseCapError(f"landmark {name} must contain three finite numbers")
    return point


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) * 0.5


def _project_hint(hint, primary, fallback, previous=None):
    projected = hint - primary * np.dot(primary, hint)
    if np.linalg.norm(projected) < max(_EPSILON, 0.05 * np.linalg.norm(hint)):
        if previous is not None:
            projected = previous - primary * np.dot(primary, previous)
        if np.linalg.norm(projected) < _EPSILON:
            projected = fallback - primary * np.dot(primary, fallback)
    return _unit(projected)


def _frame_from_xy(x_hint, y_hint, previous=None, z_hint=None):
    x = _unit(x_hint)
    fallback = np.cross(z_hint if z_hint is not None else [0, 0, 1], x)
    y = _project_hint(np.asarray(y_hint), x, fallback,
                      previous[:, 1] if previous is not None else None)
    z = _unit(np.cross(x, y))
    y = _unit(np.cross(z, x))
    if previous is not None and np.dot(y, previous[:, 1]) + np.dot(z, previous[:, 2]) < 0:
        y, z = -y, -z
    return np.column_stack((x, y, z))


def _frame_from_yx(y_hint, x_hint, previous=None, z_hint=None):
    y = _unit(y_hint)
    fallback = np.cross(y, z_hint if z_hint is not None else [0, 0, 1])
    x = _project_hint(np.asarray(x_hint), y, fallback,
                      previous[:, 0] if previous is not None else None)
    z = _unit(np.cross(x, y))
    x = _unit(np.cross(y, z))
    if previous is not None and np.dot(x, previous[:, 0]) + np.dot(z, previous[:, 2]) < 0:
        x, z = -x, -z
    return np.column_stack((x, y, z))


def _foot_frame(ankle, heel, toe):
    z = _unit(toe - heel)
    y = _project_hint(ankle - heel, z, np.array([0., 1., 0.]))
    x = _unit(np.cross(y, z))
    y = _unit(np.cross(z, x))
    return np.column_stack((x, y, z))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < _EPSILON:
        raise PoseCapError("landmarks do not define a stable anatomical frame")
    return vector / norm


def _matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    quaternion = _quaternion_from_dominant_diagonal(m)
    if trace > 0.0:
        quaternion = _quaternion_from_positive_trace(m, trace)
    return quaternion_to_axis_angle(quaternion)


def _quaternion_from_positive_trace(m: np.ndarray, trace: float) -> np.ndarray:
    scale = 2.0 * np.sqrt(trace + 1.0)
    return np.array(
        [
            0.25 * scale,
            (m[2, 1] - m[1, 2]) / scale,
            (m[0, 2] - m[2, 0]) / scale,
            (m[1, 0] - m[0, 1]) / scale,
        ]
    )


def _quaternion_from_dominant_diagonal(m: np.ndarray) -> np.ndarray:
    index = int(np.argmax(np.diag(m)))
    j = (index + 1) % 3
    k = (index + 2) % 3
    scale = 2.0 * np.sqrt(max(_EPSILON, 1.0 + m[index, index] - m[j, j] - m[k, k]))
    quaternion = np.zeros(4)
    quaternion[index + 1] = 0.25 * scale
    quaternion[0] = (m[k, j] - m[j, k]) / scale
    quaternion[j + 1] = (m[j, index] + m[index, j]) / scale
    quaternion[k + 1] = (m[k, index] + m[index, k]) / scale
    return quaternion

# MediaPipe landmark indices. Camera coordinates become canonical x-left/y-up/z-front
# through a proper rotation, not a reflection. Left/right indices are not swapped.
LANDMARK_INDICES = {
    'nose': 0, 'left_ear': 7, 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10,
    'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16, 'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26, 'left_ankle': 27, 'right_ankle': 28,
    'left_heel': 29, 'right_heel': 30, 'left_foot_index': 31, 'right_foot_index': 32,
}
CAMERA_ALIGNMENT = np.diag([1., -1., -1.])


def rotation_matrix(axis_angle):
    value = np.asarray(axis_angle, dtype=float)
    angle = np.linalg.norm(value)
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = value / angle
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


class MediaPipeBodyRetargeter:
    """Pelvis-locked, body-only motion. Target shape and camera stay fixed."""
    def __init__(self, camera_template, confidence_threshold=0.5):
        from headless_avatar_worker import validate_pose
        if not 0 <= confidence_threshold <= 1:
            raise ValueError('Invalid confidence threshold')
        self.template = validate_pose(camera_template)
        self.threshold = confidence_threshold
        self.converter = LandmarkPoseConverter()

    def convert(self, world_landmarks, confidence):
        from headless_avatar_worker import validate_pose
        points = np.asarray(world_landmarks, dtype=float)
        confidence = np.asarray(confidence, dtype=float)
        if points.shape != (33, 3) or confidence.shape != (33,):
            raise ValueError('Require 33 world landmarks and confidence values')
        indices = list(LANDMARK_INDICES.values())
        if (not np.isfinite(points).all() or not np.isfinite(confidence).all()
                or np.max(np.abs(points)) > 10 or np.min(confidence[indices]) < self.threshold):
            return None
        canonical = {name: points[index] * [1, -1, -1]
                     for name, index in LANDMARK_INDICES.items()}
        body = self.converter.convert(canonical)
        root = CAMERA_ALIGNMENT @ rotation_matrix(body['global_orient'])
        packet = {key: self.template[key] for key in ('focal', 'princpt', 'img_size_wh', 'trans')}
        packet.update(root_pose=_matrix_to_axis_angle(root).tolist(), body_pose=body['body_pose'],
                      jaw_pose=[0., 0., 0.], leye_pose=[0., 0., 0.], reye_pose=[0., 0., 0.],
                      lhand_pose=[[0., 0., 0.]] * 15, rhand_pose=[[0., 0., 0.]] * 15,
                      expr=[0.] * 100)
        return validate_pose(packet)
