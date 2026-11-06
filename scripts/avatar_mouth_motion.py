"""Opt-in video-driven jaw opening. No audio or blendshape-basis conversion.

MediaPipe blendshape scores are not FLAME expression coefficients. Only the
named jawOpen score is mapped to a bounded SMPL-X jaw X rotation here.
"""
import copy
import math


class JawOpening:
    def __init__(self, max_radians=0.40, smoothing=0.65):
        if not math.isfinite(max_radians) or not 0 < max_radians <= 0.5:
            raise ValueError('Jaw range must be in (0, 0.5] radians')
        if not math.isfinite(smoothing) or not 0 < smoothing <= 1:
            raise ValueError('Invalid smoothing')
        self.max_radians, self.smoothing = max_radians, smoothing
        self.previous = 0.0

    def apply(self, packet, score):
        if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('jawOpen must be a finite score in [0, 1]')
        target = float(score)*self.max_radians
        value = self.previous + self.smoothing*(target-self.previous)
        result = copy.deepcopy(packet)
        result['jaw_pose'] = [value, 0., 0.]
        self.previous = value
        return result

    def missing(self, packet):
        # Do not hold an open mouth indefinitely when the face disappears.
        return self.apply(packet, 0.0)
