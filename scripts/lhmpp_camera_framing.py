"""Camera calibration for an optional crop BEFORE neural rendering."""
import math


def framing_transform(old_width, old_height, target_height, box=None):
    if old_width <= 0 or old_height <= 0 or target_height < 14 or target_height % 14:
        raise ValueError('Require positive dimensions and a target height divisible by 14')
    left, top, right, bottom = box if box is not None else (0, 0, old_width, old_height)
    if not all(math.isfinite(v) for v in (left, top, right, bottom)):
        raise ValueError('Framing coordinates must be finite')
    if not (0 <= left < right <= old_width and 0 <= top < bottom <= old_height):
        raise ValueError('Framing box must lie inside the original motion image')
    width = max(14, round(target_height * (right - left) / (bottom - top) / 14) * 14)
    return {'width': width, 'height': target_height,
            'scale_x': width / (right - left), 'scale_y': target_height / (bottom - top),
            'left': left, 'top': top, 'box': [left, top, right, bottom]}


def transform_intrinsics(focal, principal, transform):
    sx, sy = transform['scale_x'], transform['scale_y']
    return ([focal[0] * sx, focal[1] * sy],
            [(principal[0] - transform['left']) * sx,
             (principal[1] - transform['top']) * sy])
