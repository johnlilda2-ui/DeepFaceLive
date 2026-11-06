"""Bounded LHM++ jaw response check before a full mouth-motion replay."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime-root', 'model-dir', 'reference', 'motion-jsonl', 'output-dir'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--framing-box', type=float, nargs=4, required=True)
    args = parser.parse_args()
    from lhmpp_avatar_backend import LHMAvatarBackend
    from headless_avatar_worker import HeadlessAvatarWorker
    import torch
    from PIL import Image, ImageDraw
    records = [json.loads(s) for s in args.motion_jsonl.read_text().splitlines() if s.strip()]
    if len(records) != 180 or [r['frame_id'] for r in records] != list(range(180)):
        raise ValueError('Require complete existing 180-frame motion')
    if not all(np.isfinite(r['pose']['jaw_pose']).all() for r in records):
        raise ValueError('Invalid jaw motion')
    source_index = max(range(180), key=lambda i:records[i]['pose']['jaw_pose'][0])
    pose = copy.deepcopy(records[source_index]['pose'])
    trials = [('Neutral', 0.), ('Tracked maximum', float(pose['jaw_pose'][0])), ('Controlled 0.4 rad', .4)]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    worker = None
    begin = time.monotonic()
    try:
        backend = LHMAvatarBackend(args.runtime_root, args.model_dir, args.reference,
                                   optimized_decoder=True, render_height=630,
                                   framing_box=args.framing_box)
        initial = copy.deepcopy(pose)
        initial['jaw_pose'] = [0., 0., 0.]
        worker = HeadlessAvatarWorker(backend, initial)
        frames = []
        for index, (label, opening) in enumerate(trials):
            sample = copy.deepcopy(pose)
            sample['jaw_pose'] = [opening, 0., 0.]
            frames.append(worker.render(index, index*100., sample))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            worker.render(3, 300., pose)
            torch.cuda.synchronize()
        kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
                   and not e.name.startswith(('Memcpy', 'Memset'))]
        if not kernels:
            raise RuntimeError('No CUDA compute evidence')
        h,w = frames[0].shape[:2]
        board = Image.new('RGB', (w*3, h+32), 'white')
        draw = ImageDraw.Draw(board)
        for index, ((label, opening), frame) in enumerate(zip(trials, frames)):
            board.paste(Image.fromarray(frame), (index*w, 32))
            draw.text((index*w+8, 8), f'{label}: {opening:.4f}', fill='black')
            Image.fromarray(frame).save(output/f'jaw-{index}.png')
        board.save(output/'comparison.png')
        changes = [np.abs(f.astype(float)-frames[0].astype(float)).mean(2) for f in frames[1:]]
        report = dict(scope='bounded_lhmpp_jaw_response', execution_passed=True,
                      visual_quality_verified=False, live_verified=False, audio_lip_sync_verified=False,
                      source_frame_id=source_index, trials=trials, output_shape=list(frames[0].shape),
                      cuda_compute_events=len(kernels), seconds=time.monotonic()-begin,
                      changed_pixels=[int((d>1).sum()) for d in changes],
                      mean_absolute_difference=[float(d.mean()) for d in changes],
                      backend=copy.deepcopy(backend.stats),
                      limitations=['Three stills from one avatar; not a lip-sync test',
                                   'Mouth opening alone cannot reproduce speech lip shapes'])
    finally:
        if worker is not None:
            worker.close()
    report['after_close'] = worker.snapshot()
    (output/'jaw-probe-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('JAW_RENDER_REPORT',json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
