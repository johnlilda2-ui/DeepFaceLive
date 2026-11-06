#!/usr/bin/env python3
"""Bounded Kaggle GPU test for the cached avatar worker, not live streaming."""
import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np

from headless_avatar_worker import HeadlessAvatarWorker, validate_pose


def read_pose(path):
    if path.stat().st_size > 65536:
        raise ValueError('Prepared pose file is too large')
    raw = json.loads(path.read_text())
    flame = path.parent.parent / 'flame_params' / path.name
    if flame.is_file():
        if flame.stat().st_size > 65536:
            raise ValueError('Prepared face pose file is too large')
        f = json.loads(flame.read_text())
        raw.update(expr=f['expcode'], jaw_pose=f['posecode'][3:6],
                   leye_pose=f['eyecode'][:3], reye_pose=f['eyecode'][3:6])
    validate_pose(raw)
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('runtime-root', 'model-dir', 'reference', 'pose-json', 'output-dir', 'baseline-dir'):
        parser.add_argument('--' + key, required=True, type=Path)
    parser.add_argument('--framing-box', type=float, nargs=4)
    parser.add_argument('--optimized-decoder', action='store_true')
    args = parser.parse_args()
    for key in ('runtime_root', 'model_dir', 'reference', 'pose_json', 'baseline_dir'):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'offline_cached_avatar_worker', 'passed': False, 'live_verified': False}
    worker = backend = None
    try:
        paths = sorted(args.pose_json.parent.glob('*.json'))
        start = paths.index(args.pose_json)
        selected = paths[start::5][:18]
        if len(selected) != 18:
            raise ValueError('Require 18 prepared poses')
        poses = [read_pose(p) for p in selected]
        from lhmpp_avatar_backend import LHMAvatarBackend
        from lhmpp_precision_quality import compare_frames
        import torch
        from PIL import Image, ImageDraw
        begin = time.monotonic()
        backend = LHMAvatarBackend(args.runtime_root, args.model_dir, args.reference,
            framing_box=args.framing_box, optimized_decoder=args.optimized_decoder)
        worker = HeadlessAvatarWorker(backend, poses[0])
        torch.cuda.synchronize()
        report['setup_seconds'] = time.monotonic() - begin
        backend.source.save(output / 'reference.png')
        counter = 0
        def sequence():
            nonlocal counter
            frames = []
            for pose in poses:
                frames.append(worker.render(counter, counter * 1000 / 6, pose))
                counter += 1
            return np.stack(frames)
        with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            verified = sequence()
            torch.cuda.synchronize()
        kernels = [e.name for e in prof.events()
                   if e.device_type == torch.autograd.DeviceType.CUDA
                   and not e.name.startswith(('Memcpy', 'Memset'))]
        prof.export_chrome_trace(str(output / 'render-cuda-trace.json'))
        if not kernels:
            raise RuntimeError('Worker rendering executed no CUDA compute kernels')
        report['render_cuda_compute_kernel_events'] = len(kernels)
        report['render_cuda_kernel_examples'] = kernels[:8]
        sequence()  # warmup
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(3):
            torch.cuda.synchronize()
            begin = time.monotonic()
            frames = sequence()
            torch.cuda.synchronize()
            seconds = time.monotonic() - begin
            mae = float(np.abs(frames.astype(float) - verified.astype(float)).mean())
            if mae > 0.5:
                raise RuntimeError('Cached worker changed output across repeats')
            samples.append({'sequence_seconds': seconds, 'fps': 18 / seconds,
                            'mean_abs_difference_from_verified_run': mae})
        report.update(benchmark_repeats=samples, median_fps=float(np.median([s['fps'] for s in samples])),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        baseline = np.load(args.baseline_dir / 'frames.npy', allow_pickle=False)
        quality = compare_frames(baseline, verified)
        report['baseline_quality'] = quality
        if not quality['quality_gate_passed']:
            raise RuntimeError('Worker output differs excessively from the verified probe')
        before = worker.snapshot()['backend']['render_calls']
        try:
            worker.render(counter - 1, (counter - 1) * 1000 / 6, poses[-1])
            raise AssertionError('Stale frame was accepted')
        except ValueError:
            report['stale_frame_rejected'] = True
        invalid = dict(poses[0]); invalid['focal'] = [float('nan'), 1000]
        try:
            worker.render(counter, counter * 1000 / 6, invalid)
            raise AssertionError('Invalid motion was accepted')
        except ValueError:
            report['invalid_frame_rejected'] = True
        if worker.snapshot()['backend']['render_calls'] != before:
            raise RuntimeError('Rejected packet reached the GPU backend')
        changed_shape = dict(poses[0]); changed_shape['betas'] = [999.0] * 10
        identity_test = worker.render(counter, counter * 1000 / 6, changed_shape)
        shape_error = float(np.abs(identity_test.astype(float) - verified[0].astype(float)).mean())
        if shape_error > 0.5:
            raise RuntimeError('Incoming shape coefficients changed avatar identity')
        report['incoming_shape_override_mae'] = shape_error
        report['before_close'] = worker.snapshot()
        if backend.stats['model_loads'] != 1 or backend.stats['avatar_reconstructions'] != 1:
            raise RuntimeError('Avatar was loaded or reconstructed more than once')
        np.save(output / 'frames.npy', verified, allow_pickle=False)
        Image.fromarray(verified[0]).save(output / 'frame.png')
        import imageio.v3 as iio
        iio.imwrite(output / 'motion.mp4', verified, fps=6, codec='libx264',
                    pixelformat='yuv420p', macro_block_size=2)
        import subprocess
        codec = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
            '-select_streams', 'v:0', '-show_entries',
            'stream=codec_name,width,height,r_frame_rate,duration,nb_read_frames',
            '-of', 'json', str(output / 'motion.mp4')], text=True, timeout=30))['streams'][0]
        if (codec['codec_name'] != 'h264' or int(codec['nb_read_frames']) != 18
                or codec['width'] != verified.shape[2] or codec['height'] != verified.shape[1]
                or codec['r_frame_rate'] != '6/1' or abs(float(codec['duration']) - 3) > 0.01):
            raise RuntimeError('Encoded clip verification failed')
        report['codec_audit'] = codec
        h, w = verified.shape[1:3]
        board = Image.new('RGB', (w * 3, h * 2 + 60), 'white')
        draw = ImageDraw.Draw(board)
        draw.text((10, 10), 'Probe baseline (top) | Persistent worker (bottom) | Same fictional avatar', fill='black')
        for row, values in enumerate((baseline, verified)):
            for col, i in enumerate((0, 5, 17)):
                x, y = col * w, 45 + row * (h + 5)
                board.paste(Image.fromarray(values[i]), (x, y))
                draw.text((x + 5, y), f'Frame {i + 1}', fill='black')
        board.save(output / 'comparison.png')
        report['first_last_frame_mae'] = float(np.abs(verified[0].astype(float) - verified[-1].astype(float)).mean())
        if report['first_last_frame_mae'] < 0.1:
            raise RuntimeError('Worker output was static')
        worker.close()
        worker.close()
        try:
            worker.render(counter + 1, (counter + 1) * 1000 / 6, poses[0])
            raise AssertionError('Closed worker accepted a frame')
        except RuntimeError:
            report['render_after_close_rejected'] = True
        report['after_close'] = worker.snapshot()
        stats = report['after_close']['backend']
        if (stats['cache_present'] or not stats.get('model_python_released', False)
                or report['after_close']['close_error'] is not None
                or stats['allocated_after_close'] > stats['allocated_before_load'] + 1024**2):
            raise RuntimeError('Avatar GPU state was not released')
        report['passed'] = True
    except Exception:
        report['error'] = traceback.format_exc()
    finally:
        if worker is not None:
            worker.close()
            report['final_state'] = worker.snapshot()
        elif backend is not None:
            backend.close()
            report['final_backend'] = backend.stats
    (output / 'worker-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
