#!/usr/bin/env python3
"""Replay CPU-retargeted packets through the isolated cached avatar worker."""
import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np

from headless_avatar_worker import HeadlessAvatarWorker, validate_pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime-root', 'model-dir', 'reference', 'motion-jsonl', 'driver-video', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--driver-kind', choices=('synthetic', 'consented-human'), default='synthetic',
                        help='Record the supplied clip provenance; this does not grant consent')
    args = parser.parse_args()
    for name in ('runtime_root', 'model_dir', 'reference', 'motion_jsonl', 'driver_video'):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'offline_tracked_motion_to_avatar', 'passed': False, 'live_verified': False,
              'driver_kind': args.driver_kind}
    worker = None
    try:
        if args.motion_jsonl.stat().st_size > 1024**2:
            raise ValueError('Bounded test motion file is too large')
        records = [json.loads(line) for line in args.motion_jsonl.read_text().splitlines() if line.strip()]
        if not 15 <= len(records) <= 18:
            raise ValueError('Require at least 15 valid poses from the 18-frame tracking test')
        for r in records:
            validate_pose(r['pose'])
        from lhmpp_avatar_backend import LHMAvatarBackend
        import torch
        import cv2
        import imageio.v3 as iio
        from PIL import Image, ImageDraw, ImageOps
        begin = time.monotonic()
        backend = LHMAvatarBackend(args.runtime_root, args.model_dir, args.reference,
                                   optimized_decoder=True)
        worker = HeadlessAvatarWorker(backend, records[0]['pose'])
        torch.cuda.synchronize()
        report['setup_seconds'] = time.monotonic() - begin
        frames, bounds = [], []
        begin = time.monotonic()
        with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            for r in records:
                frame = worker.render(r['frame_id'], r['timestamp_ms'], r['pose'])
                frames.append(frame)
                y, x = np.where(frame.min(axis=-1) < 230)
                if len(x) == 0:
                    raise RuntimeError('Retargeting produced a blank frame')
                bounds.append([int(x.min()), int(y.min()), int(x.max()), int(y.max())])
            torch.cuda.synchronize()
        report['profiled_render_seconds'] = time.monotonic() - begin
        kernels = [e.name for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA and not e.name.startswith(('Memcpy', 'Memset'))]
        prof.export_chrome_trace(str(output / 'render-cuda-trace.json'))
        if not kernels:
            raise RuntimeError('No actual CUDA rendering kernels')
        frames = np.stack(frames)
        np.save(output / 'frames.npy', frames, allow_pickle=False)
        source = []
        cap = cv2.VideoCapture(str(args.driver_video))
        try:
            for i in range(18):
                ok, image = cap.read()
                if not ok: break
                source.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        if len(source) != 18:
            raise RuntimeError('Driving fixture did not decode 18 frames')
        by_id = {r['frame_id']: frame for r, frame in zip(records, frames)}
        timeline, held = [], []
        last = np.full_like(frames[0], 255)
        for i in range(18):
            if i in by_id: last = by_id[i]
            else: held.append(i)
            timeline.append(last)
        iio.imwrite(output / 'motion.mp4', np.stack(timeline), fps=6, codec='libx264',
                    pixelformat='yuv420p', macro_block_size=2)
        h, w = frames.shape[1:3]
        edge = [r['frame_id'] for r, (l,t,rr,b) in zip(records, bounds)
                if l < 3 or t < 3 or rr >= w-3 or b >= h-3]
        variation = float(np.mean(np.abs(frames[-1].astype(float)-frames[0].astype(float))))
        board = Image.new('RGB', (950, 990), 'white')
        draw = ImageDraw.Draw(board)
        draw.text((12, 10), 'Generated driving clip (left) | CPU-tracked avatar motion (right)', fill='black')
        for row, i in enumerate((0, len(records)//2, len(records)-1)):
            r = records[i]
            for image, left, width in [(Image.fromarray(source[r['frame_id']]),0,300),
                                        (Image.fromarray(frames[i]),310,630)]:
                tile = ImageOps.contain(image, (width, 280))
                board.paste(tile, (left+(width-tile.width)//2, 50+row*310))
            draw.text((12, 335+row*310), f"Driving frame {r['frame_id']+1}", fill='black')
        board.save(output / 'comparison.png')
        report.update(rendered_frames=len(records), frame_ids=[r['frame_id'] for r in records],
            held_timeline_frames=held, output_shape=list(frames[0].shape),
            cuda_compute_kernel_events=len(kernels), foreground_bounds=bounds,
            edge_contact_frames=edge, first_last_frame_mae=variation,
            before_close=worker.snapshot(),
            limitations=[('Synthetic driving clip only' if args.driver_kind == 'synthetic'
                           else 'Short consented-human driving clip only'),'Pelvis-locked body motion',
                         'Approximate limb twist; neutral fingers and facial expressions',
                         'Profiled render time is not an end-to-end live FPS measurement'])
        worker.close()
        report['after_close'] = worker.snapshot()
        stats = report['after_close']['backend']
        if (stats['avatar_reconstructions'] != 1 or not stats['model_python_released']
                or stats['cache_present'] or report['after_close']['close_error'] is not None
                or stats['allocated_after_close'] > 1024**2 or edge or variation < 0.1):
            raise RuntimeError('Avatar motion, framing or cleanup screening failed')
        report['passed'] = True
    except Exception:
        report['error'] = traceback.format_exc()
    finally:
        if worker is not None: worker.close()
    (output / 'replay-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
