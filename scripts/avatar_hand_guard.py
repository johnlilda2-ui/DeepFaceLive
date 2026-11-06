"""Optional rejection of abrupt finger observations; not image reconstruction."""
import math
import numpy as np
from avatar_head_hands import FINGERS, points21, palm_frame


class HandObservationGuard:
    def __init__(self):
        self.previous=None
        self.timestamp=None

    def reset(self):
        self.previous=self.timestamp=None

    def check(self, points, timestamp_ms):
        if (type(timestamp_ms) not in (float,int) or not math.isfinite(timestamp_ms) or timestamp_ms<0 or
                self.timestamp is not None and timestamp_ms<=self.timestamp):
            raise ValueError('Hand observations need increasing timestamps')
        points=points21(points)
        canonical=(points-points[0]) @ palm_frame(points)
        bones=np.array([canonical[b]-canonical[a] for chain in FINGERS for a,b in zip(chain[:-1],chain[1:])])
        lengths=np.linalg.norm(bones,axis=1)
        directions=bones/lengths[:,None]
        if self.previous is not None and timestamp_ms-self.timestamp<500:
            old_directions,old_lengths=self.previous
            ratio=lengths/old_lengths
            if np.min(ratio)<.5 or np.max(ratio)>2:
                return False,'segment_length_jump'
            angle=np.arccos(np.clip((directions*old_directions).sum(1),-1,1))
            limit=math.radians(min(150.,25.+.6*(timestamp_ms-self.timestamp)))
            if angle.max()>limit:
                return False,'finger_direction_jump'
        self.previous=(directions.copy(),lengths.copy())
        self.timestamp=timestamp_ms
        return True,'accepted'
