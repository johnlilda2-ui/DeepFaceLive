"""Opt-in, fitted ARKit lip controls to FLAME expression coefficients.

The fit is computed from the operator's prepared licensed model assets. This
module never guesses coefficient indices or downloads/distributes model data.
"""
import copy
import json
import math
from pathlib import Path
import numpy as np

LIP_CONTROLS = ('mouthPucker', 'mouthFunnel', 'mouthSmileLeft', 'mouthSmileRight',
                'mouthPressLeft', 'mouthPressRight', 'mouthRollLower',
                'mouthRollUpper', 'mouthUpperUpLeft', 'mouthUpperUpRight',
                'mouthLowerDownLeft', 'mouthLowerDownRight')


def fit_lip_basis(expression_dirs, lip_offsets, names, maximum_relative_error=.45):
    dirs = np.asarray(expression_dirs, dtype=float)
    offsets = np.asarray(lip_offsets, dtype=float)
    names = list(names)
    if (dirs.ndim != 3 or dirs.shape[1:] != (3,100) or
            offsets.shape != (len(names), dirs.shape[0], 3) or
            len(set(names)) != len(names) or any(n not in LIP_CONTROLS for n in names) or
            not np.isfinite(dirs).all() or not np.isfinite(offsets).all() or
            not 0 < maximum_relative_error < 1):
        raise ValueError('Invalid lip/FLAME basis layout')
    design = dirs.reshape(-1,100)
    target = offsets.reshape(len(names),-1).T
    # Ridge scale follows the basis, rather than assuming millimetres/metres.
    gram = design.T @ design
    ridge = max(float(np.trace(gram))/100*1e-5, 1e-12)
    projection = np.linalg.solve(gram+np.eye(100)*ridge, design.T @ target)
    norms = np.linalg.norm(target, axis=0)
    error = np.linalg.norm(design @ projection-target, axis=0)/np.maximum(norms, 1e-12)
    accepted = (norms>1e-8) & (error <= maximum_relative_error)
    # Unsupported controls remain explicitly rejected; never silently weaken gate.
    projection[:,~accepted] = 0
    return projection, dict(relative_errors=error.tolist(), accepted=accepted.tolist(),
                            maximum_relative_error=maximum_relative_error,
                            ridge=ridge, names=names)


class LipExpressions:
    def __init__(self, projection, metadata, tau_ms=60., max_norm=6.):
        self.projection = np.asarray(projection, dtype=float).copy()
        self.names = metadata['names']
        accepted = metadata['accepted']
        if (self.projection.shape != (100,len(self.names)) or
                len(set(self.names)) != len(self.names) or
                any(n not in LIP_CONTROLS for n in self.names) or
                len(accepted) != len(self.names) or
                not all(type(v) is bool for v in accepted) or
                not np.isfinite(self.projection).all() or
                np.max(np.abs(self.projection),initial=0)>100 or
                not math.isfinite(tau_ms) or not tau_ms>0 or
                not math.isfinite(max_norm) or not 0<max_norm<=6):
            raise ValueError('Invalid fitted expression map')
        if any(np.any(self.projection[:,i]) for i,v in enumerate(accepted) if not v):
            raise ValueError('Rejected lip controls must have zero projection')
        self.tau_ms, self.max_norm = tau_ms, max_norm
        self.previous = np.zeros(100)
        self.last_timestamp = None

    @classmethod
    def load(cls, path):
        if Path(path).stat().st_size > 256*1024:
            raise ValueError('Expression map too large')
        with np.load(path,allow_pickle=False) as data:
            return cls(data['projection'],json.loads(str(data['metadata'].item())))

    def apply(self, packet, scores, timestamp_ms):
        if not isinstance(scores,dict):
            raise ValueError('Require named blendshape scores')
        if (type(timestamp_ms) not in (float,int) or not math.isfinite(timestamp_ms) or
                timestamp_ms<0 or (self.last_timestamp is not None and timestamp_ms<=self.last_timestamp)):
            raise ValueError('Require increasing finite media timestamps')
        weights = []
        for name in self.names:
            value = scores.get(name,0.)
            if type(value) not in (float,int) or not math.isfinite(value) or not 0<=value<=1:
                raise ValueError('Invalid named lip score')
            weights.append(value)
        target = self.projection @ weights
        norm = np.linalg.norm(target)
        if norm>self.max_norm:
            target *= self.max_norm/norm
        dt = self.tau_ms if self.last_timestamp is None else timestamp_ms-self.last_timestamp
        alpha = 1-math.exp(-min(dt,1000.)/self.tau_ms)
        value = self.previous + alpha*(target-self.previous)
        base = np.asarray(packet.get('expr',np.zeros(100)),dtype=float)
        if base.shape!=(100,) or not np.isfinite(base).all():
            raise ValueError('Invalid baseline expression')
        result = copy.deepcopy(packet)
        combined = base+value
        norm = np.linalg.norm(combined)
        if norm>self.max_norm:
            combined *= self.max_norm/norm
        result['expr'] = combined.tolist()
        self.previous, self.last_timestamp = value, timestamp_ms
        return result
