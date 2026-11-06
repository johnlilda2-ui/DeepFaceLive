#!/usr/bin/env python3
"""Audit two completed offline allocator probes; never claims live readiness."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw


def audit(control_dir, reuse_dir, output_dir):
    control_dir, reuse_dir = control_dir.resolve(), reuse_dir.resolve()
    if control_dir == reuse_dir:
        raise ValueError('Control and optimized outputs must be distinct')
    reports = [json.loads((p / 'report.json').read_text()) for p in (control_dir, reuse_dir)]
    for report in reports:
        if not report.get('passed') or not report.get('strict_weights_loaded') or report.get('cuda_kernel_count', 0) <= 0:
            raise ValueError('Both runs must pass strict loading and CUDA verification')
        bench = report['unprofiled_cached_benchmark']
        if bench['warmup_sequences'] != 1 or len(bench['repeats']) != 3:
            raise ValueError('Require one warmup and three timed repeats')
        fps = [sample['fps_including_cpu_and_readback'] for sample in bench['repeats']]
        if not all(np.isfinite(fps)) or min(fps) <= 0:
            raise ValueError('Invalid measured throughput')
        if not np.isclose(np.median(fps), bench['median_fps_including_cpu_and_readback']):
            raise ValueError('Reported median does not match repeats')
    for key in ('gpu', 'upstream_commit', 'reference', 'reference_betas', 'motion_frame_files', 'camera_framing', 'random_seed'):
        if reports[0][key] != reports[1][key]:
            raise ValueError('A/B input mismatch: ' + key)
    if 'allocator_adapter' in reports[0] or reports[1]['allocator_adapter']['suppressed_empty_cache_calls'] <= 0:
        raise ValueError('Missing allocator ablation')
    arrays = [np.load(p / 'frames.npy', allow_pickle=False) for p in (control_dir, reuse_dir)]
    expected = (18, *reports[0]['frame_shape'])
    if any(a.shape != expected or a.dtype != np.uint8 for a in arrays):
        raise ValueError('Unexpected raw sequence shape or dtype')
    difference = np.abs(arrays[0].astype(np.int16) - arrays[1].astype(np.int16))
    # Keep the exact-equality gate strict, but preserve failed experiment evidence.
    pixels_equal = bool(difference.max() == 0)
    codec_audits = []
    for path, report in zip((control_dir, reuse_dir), reports):
        result = subprocess.run(['ffprobe', '-v', 'error', '-count_frames',
            '-select_streams', 'v:0', '-show_entries',
            'stream=codec_name,width,height,r_frame_rate,duration,nb_read_frames',
            '-of', 'json', str(path / 'motion.mp4')],
            check=True, capture_output=True, text=True, timeout=30)
        stream = json.loads(result.stdout)['streams'][0]
        height, width, _ = report['frame_shape']
        if (stream['codec_name'] != 'h264' or int(stream['nb_read_frames']) != 18
                or stream['width'] != width or stream['height'] != height
                or stream['r_frame_rate'] != '6/1' or abs(float(stream['duration']) - 3) > 0.01):
            raise ValueError('Encoded clip does not match the bounded 18-frame test')
        codec_audits.append(stream)
    a, b = [r['unprofiled_cached_benchmark']['median_fps_including_cpu_and_readback'] for r in reports]
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        'scope': 'offline_cached_allocator_ablation', 'passed': pixels_equal, 'live_verified': False,
        'rejection_reasons': [] if pixels_equal else ['Exact raw-pixel equality failed'],
        'gpu': reports[0]['gpu'], 'control_directory': str(control_dir), 'reuse_directory': str(reuse_dir),
        'control_benchmark': reports[0]['unprofiled_cached_benchmark'],
        'reuse_benchmark': reports[1]['unprofiled_cached_benchmark'],
        'speedup_percent': (b / a - 1) * 100,
        'raw_frames_equal': pixels_equal,
        'changed_channel_values': int(np.count_nonzero(difference)),
        'total_channel_values': int(difference.size), 'raw_pixel_max_error': int(difference.max()),
        'raw_pixel_mean_error': float(difference.mean()), 'compared_frames': 18,
        'codec_audits': codec_audits,
        'cuda_compute_kernel_events': [r['cuda_kernel_count'] for r in reports],
        'adapter': reports[1]['allocator_adapter'],
        'limitations': ['One fictional reference and 18 prerecorded poses',
                        'Tracking, network, encoding and reference reconstruction excluded from FPS',
                        'No improvement to underlying identity or hand/face quality is claimed']
    }
    (output_dir / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    # Same pose pairs allow visual inspection, without rescaling either condition.
    h, w = arrays[0].shape[1:3]
    canvas = Image.new('RGB', (w * 3, h * 2 + 70), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f'Control {a:.2f} FPS | Reuse {b:.2f} FPS | Raw max difference: {int(difference.max())}/255', fill='black')
    for row, frames in enumerate(arrays):
        for col, idx in enumerate((0, 5, 17)):
            x, y = col * w, 55 + row * (h + 5)
            canvas.paste(Image.fromarray(frames[idx]), (x, y))
            draw.text((x + 5, y), f'{"Control" if row == 0 else "Reuse"} frame {idx + 1}', fill='black')
    canvas.save(output_dir / 'comparison.png')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-dir', required=True, type=Path)
    parser.add_argument('--reuse-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.control_dir, args.reuse_dir, args.output_dir)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
