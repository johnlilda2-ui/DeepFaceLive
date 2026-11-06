"""Predeclared image-quality gates for the offline decoder precision experiment."""
import numpy as np

# Approximate arithmetic has explicit tolerances; the allocator equality gate
# remains separate and strict. These are screening criteria, not likeness proof.
LIMITS = {'full_mae': 1.0, 'subject_mae': 2.0, 'subject_p99': 8.0,
          'subject_large_error_fraction': 0.001, 'minimum_frame_iou': 0.99,
          'maximum_frame_subject_mae': 2.0, 'minimum_speedup_percent': 20.0}


def compare_frames(reference, candidate):
    if (reference.shape != candidate.shape or reference.ndim != 4
            or reference.shape[-1] != 3 or reference.dtype != np.uint8
            or candidate.dtype != np.uint8 or len(reference) == 0):
        raise ValueError('Require matching nonempty uint8 RGB sequences')
    error = np.abs(reference.astype(np.float32) - candidate.astype(np.float32))
    base_masks = reference.min(axis=-1) < 230
    new_masks = candidate.min(axis=-1) < 230
    subject = base_masks | new_masks
    if any(not m.any() for m in base_masks):
        raise ValueError('Every baseline frame needs visible foreground')
    per_frame_mae = [float(e[m].mean()) for e, m in zip(error, subject)]
    ious = [float((a & b).sum() / (a | b).sum())
            for a, b in zip(base_masks, new_masks)]
    selected = error[subject]
    metrics = {'full_mae': float(error.mean()), 'subject_mae': float(selected.mean()),
        'subject_p99': float(np.percentile(selected, 99)),
        'subject_large_error_fraction': float((selected > 16).mean()),
        'minimum_frame_iou': min(ious),
        'maximum_frame_subject_mae': max(per_frame_mae),
        'frame_subject_mae': per_frame_mae, 'frame_iou': ious,
        'maximum_channel_error': float(error.max()),
        'mask_method': 'union of thresholded nonwhite pixels; not semantic segmentation'}
    reasons = []
    for key in ('full_mae', 'subject_mae', 'subject_p99', 'subject_large_error_fraction',
                'maximum_frame_subject_mae'):
        if metrics[key] > LIMITS[key]:
            reasons.append(key + ' exceeded tolerance')
    if metrics['minimum_frame_iou'] < LIMITS['minimum_frame_iou']:
        reasons.append('foreground overlap below tolerance')
    metrics['quality_gate_passed'] = not reasons
    metrics['rejection_reasons'] = reasons
    return metrics
