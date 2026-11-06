#!/usr/bin/env python3
"""Audit an offline FP32/decoder-FP16 A/B run against fixed screening limits."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw
from lhmpp_precision_quality import LIMITS, compare_frames


def audit(control_dir, candidate_dir, output_dir):
    paths = [control_dir.resolve(), candidate_dir.resolve()]
    if paths[0] == paths[1]:
        raise ValueError('Require distinct control and candidate directories')
    reports = [json.loads((p / 'report.json').read_text()) for p in paths]
    for r in reports:
        if not r.get('passed') or not r.get('strict_weights_loaded') or r.get('cuda_kernel_count', 0) <= 0:
            raise ValueError('Both runs need strict loading and executed CUDA kernels')
        if 'allocator_adapter' in r:
            raise ValueError('Allocator behavior must remain unchanged for this ablation')
        b = r['unprofiled_cached_benchmark']
        samples = b['repeats']
        if b['warmup_sequences'] != 1 or len(samples) != 3:
            raise ValueError('Require one warmup and three measured repeats')
        fps = [s['fps_including_cpu_and_readback'] for s in samples]
        if not np.isfinite(fps).all() or min(fps) <= 0:
            raise ValueError('Invalid timing sample')
        if not np.isclose(np.median(fps), b['median_fps_including_cpu_and_readback']):
            raise ValueError('Incorrect median')
    for key in ('gpu', 'upstream_commit', 'reference', 'reference_betas',
                'motion_frame_files', 'camera_framing', 'random_seed'):
        if reports[0][key] != reports[1][key]:
            raise ValueError('A/B inputs differ: ' + key)
    if 'decoder_precision' in reports[0]:
        raise ValueError('Control must use original decoder precision')
    precision = reports[1]['decoder_precision']
    if precision['scope'] != 'neural_decoder_only' or 'torch.float16' not in precision['observed_output_dtypes']:
        raise ValueError('Actual decoder FP16 output evidence is missing')
    refs = [np.asarray(Image.open(p / 'reference.png').convert('RGB')) for p in paths]
    if not np.array_equal(*refs):
        raise ValueError('Reference pixels differ')
    arrays = [np.load(p / 'frames.npy', allow_pickle=False) for p in paths]
    if any(a.shape != (18, *r['frame_shape']) for a, r in zip(arrays, reports)):
        raise ValueError('Unexpected sequence shape')
    quality = compare_frames(*arrays)
    codec_audits = []
    for p, r in zip(paths, reports):
        run = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name,width,height,r_frame_rate,duration,nb_read_frames',
            '-of', 'json', str(p / 'motion.mp4')],
            check=True, capture_output=True, text=True, timeout=30)
        info = json.loads(run.stdout)['streams'][0]
        h, w, _ = r['frame_shape']
        if (info['codec_name'] != 'h264' or int(info['nb_read_frames']) != 18
                or info['width'] != w or info['height'] != h or info['r_frame_rate'] != '6/1'
                or abs(float(info['duration']) - 3) > 0.01):
            raise ValueError('Encoded video audit failed')
        codec_audits.append(info)
    a, b = [r['unprofiled_cached_benchmark']['median_fps_including_cpu_and_readback'] for r in reports]
    speedup = (b / a - 1) * 100
    reasons = quality['rejection_reasons'].copy()
    if speedup < LIMITS['minimum_speedup_percent']:
        reasons.append('Speed gain below 20 percent screening target')
    result = {'scope': 'offline_decoder_precision_ablation',
        'automatic_gate_passed': not reasons, 'visual_review_required': True, 'live_verified': False,
        'rejection_reasons': reasons, 'limits': LIMITS, 'quality': quality,
        'control_directory': str(paths[0]), 'candidate_directory': str(paths[1]),
        'control_benchmark': reports[0]['unprofiled_cached_benchmark'],
        'candidate_benchmark': reports[1]['unprofiled_cached_benchmark'],
        'speedup_percent': speedup, 'decoder_precision': precision,
        'cuda_compute_kernel_events': [r['cuda_kernel_count'] for r in reports],
        'codec_audits': codec_audits}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    h, w = arrays[0].shape[1:3]
    board = Image.new('RGB', (w * 3, h * 2 + 70), 'white')
    draw = ImageDraw.Draw(board)
    draw.text((10, 10), f'Control {a:.2f} FPS | Decoder FP16 {b:.2f} FPS | Gain {speedup:.1f}%', fill='black')
    for row, frames in enumerate(arrays):
        for col, i in enumerate((0, 5, 17)):
            x, y = col * w, 55 + row * (h + 5)
            board.paste(Image.fromarray(frames[i]), (x, y))
            draw.text((x + 5, y), f'{"FP32" if row == 0 else "Decoder FP16"} frame {i + 1}', fill='black')
    board.save(output_dir / 'comparison.png')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('control-dir', 'candidate-dir', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.control_dir, args.candidate_dir, args.output_dir)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['automatic_gate_passed'] else 1)
