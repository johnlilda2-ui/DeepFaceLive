"""Opt-in neutral head calibration and geometric SMPL-X hand retargeting.

Does not alter the body-only adapter. Rest-hand coordinates are read from the
prepared licensed SMPL-X asset at runtime, not distributed in this repository.
"""
import copy
import numpy as np
from avatar_body_retarget import (CAMERA_ALIGNMENT, _frame_from_xy,
                                 _matrix_to_axis_angle, rotation_matrix)

BODY_PARENTS = (0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19)
FINGERS = ((5,6,7,8),(9,10,11,12),(17,18,19,20),
           (13,14,15,16),(1,2,3,4))  # SMPL-X index,middle,pinky,ring,thumb


class NeutralHeadCalibration:
    """Explicit neutral reference; it must be captured facing forward and level."""
    def __init__(self):
        self.reference = None

    def calibrate(self, packet):
        self.reference = rotation_matrix(packet['body_pose'][14]).copy()

    def apply(self, packet):
        if self.reference is None:
            raise ValueError('Capture an explicit neutral-head reference first')
        result = copy.deepcopy(packet)
        measured = rotation_matrix(packet['body_pose'][14])
        result['body_pose'][14] = _matrix_to_axis_angle(measured @ self.reference.T).tolist()
        return result


def body_global_rotations(packet):
    result = [CAMERA_ALIGNMENT.T @ rotation_matrix(packet['root_pose'])]
    for index, parent in enumerate(BODY_PARENTS):
        result.append(result[parent] @ rotation_matrix(packet['body_pose'][index]))
    return result


def points21(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (21,3) or not np.isfinite(value).all():
        raise ValueError('Require 21 finite 3D hand points')
    centered = value - value[0]
    if np.max(np.linalg.norm(centered, axis=1)) > 0.5:
        raise ValueError('Hand coordinates exceed a bounded anatomical extent')
    for chain in FINGERS:
        lengths = np.linalg.norm(np.diff(value[list(chain)], axis=0), axis=1)
        if np.min(lengths) < 0.001 or np.max(lengths) > 0.12:
            raise ValueError('Degenerate or implausible finger segment')
    return value


def palm_frame(points):
    return _frame_from_xy(points[9] - points[0], points[5] - points[17])


def blend_rotation(previous, target, amount):
    return previous @ rotation_matrix(
        _matrix_to_axis_angle(previous.T @ target) * amount)


class HandRetargeter:
    """Match palm and finger segment directions using the target rest geometry."""
    def __init__(self, rest_points, side, smoothing=0.6):
        if side not in ('left','right') or not 0 < smoothing <= 1:
            raise ValueError('Invalid side or smoothing')
        self.side = side
        self.rest = points21(rest_points).copy()
        self.rest_palm = palm_frame(self.rest)
        self.rest_frames = []
        for chain in FINGERS:
            self.rest_frames.append([
                _frame_from_xy(self.rest[b]-self.rest[a], self.rest_palm[:,1],
                               z_hint=self.rest_palm[:,2])
                for a,b in zip(chain[:-1],chain[1:])])
        self.smoothing = smoothing
        self.previous = None

    def reset(self):
        self.previous = None

    def apply(self, packet, canonical_points):
        points = points21(canonical_points)
        palm = palm_frame(points)
        wrist = palm @ self.rest_palm.T
        local_hands = []
        for chain, rest_frames in zip(FINGERS, self.rest_frames):
            parent = wrist
            for a,b,rest_frame in zip(chain[:-1],chain[1:],rest_frames):
                frame = _frame_from_xy(points[b]-points[a], palm[:,1], z_hint=palm[:,2])
                joint = frame @ rest_frame.T
                local_hands.append(parent.T @ joint)
                parent = joint
        matrices = [wrist] + local_hands
        if self.previous is not None:
            matrices = [blend_rotation(p, n, self.smoothing)
                        for p,n in zip(self.previous, matrices)]
        # Commit temporal state only after all geometry has passed validation.
        self.previous = [m.copy() for m in matrices]
        result = copy.deepcopy(packet)
        body = body_global_rotations(packet)
        elbow_joint, wrist_body = (18,19) if self.side == 'left' else (19,20)
        result['body_pose'][wrist_body] = _matrix_to_axis_angle(
            body[elbow_joint].T @ matrices[0]).tolist()
        result['lhand_pose' if self.side == 'left' else 'rhand_pose'] = [
            _matrix_to_axis_angle(m).tolist() for m in matrices[1:]]
        return result
