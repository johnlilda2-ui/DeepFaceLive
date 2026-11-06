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
    ideal_width = target_height * (right - left) / (bottom - top)
    # A cropped virtual camera keeps one uniform scale so the character's body
    # proportions do not change when width is rounded to the model's patch size.
    width = max(14, (math.ceil(ideal_width / 14) if box is not None
                     else round(ideal_width / 14)) * 14)
    scale_y = target_height / (bottom - top)
    scale_x = scale_y if box is not None else width / (right - left)
    pad_x = (width - ideal_width) / 2 if box is not None else 0
    return {'width': width, 'height': target_height,
            'scale_x': scale_x, 'scale_y': scale_y, 'pad_x': pad_x,
            'left': left, 'top': top, 'box': [left, top, right, bottom]}


def transform_intrinsics(focal, principal, transform):
    sx, sy = transform['scale_x'], transform['scale_y']
    return ([focal[0] * sx, focal[1] * sy],
            [(principal[0] - transform['left']) * sx + transform['pad_x'],
             (principal[1] - transform['top']) * sy])
