"""Bounded same-cache renderer diagnostic; this does not claim hand quality."""
import argparse
import json
from pathlib import Path
import time
import traceback
import hashlib
import numpy as np
from PIL import Image, ImageDraw
from headless_avatar_worker import validate_pose
from bounded_motion_test import validate_records
from lhmpp_avatar_backend import LHMAvatarBackend

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('runtime-root','model-dir','reference','motion-jsonl','baseline-frames','output-dir'):
        p.add_argument('--'+key,required=True,type=Path)
    args=p.parse_args()
    output=args.output_dir;output.mkdir(parents=True,exist_ok=False)
    report={'scope':'same_cache_hand_renderer_diagnostic','execution_passed':False,
            'hand_quality_verified':False,'live_verified':False}
    backend=None
    try:
        records=[json.loads(x) for x in args.motion_jsonl.read_text().splitlines() if x.strip()]
        validate_records(records,180)
        poses={r['frame_id']:validate_pose(r['pose']) for r in records}
        ids=[7,24,48]
        if any(i not in poses for i in [0]+ids):
            raise ValueError('Diagnostic requires the selected validated source frames')
        baseline=np.load(args.baseline_frames,mmap_mode='r',allow_pickle=False)
        if baseline.shape!=(180,630,518,3) or baseline.dtype!=np.uint8:
            raise ValueError('Require the matching 180-frame full-view baseline')
        import torch
        # Same world-space rectangle as baseline pixels x=30..488, y=80..330.
        # Crop test only: it cannot replace a full-body output.
        backend=LHMAvatarBackend(args.runtime_root,args.model_dir,args.reference,
            optimized_decoder=True,render_height=420,
            framing_box=[465,223,998,514])
        backend.prepare(poses[0])
        cache_id=id(backend.cache);model_id=id(backend.model)
        pairs=[]
        started=time.monotonic()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            for i in ids:
                row=[]
                for mode in ('neural','gs'):
                    frame=backend.render(poses[i],output_renderer=mode)
                    if id(backend.cache)!=cache_id or id(backend.model)!=model_id:
                        raise RuntimeError('Reconstruction changed between renderer probes')
                    Image.fromarray(frame).save(output/f'frame-{i}-{mode}.png')
                    row.append(frame)
                pairs.append(row)
            torch.cuda.synchronize()
        kernels=[e.name for e in prof.events()
                 if e.device_type==torch.autograd.DeviceType.CUDA
                 and not e.name.startswith(('Memcpy','Memset'))]
        if not kernels:raise RuntimeError('No CUDA compute events')
        h,w=pairs[0][0].shape[:2]
        board=Image.new('RGB',(3*w,3*(h+32)+50),'white');draw=ImageDraw.Draw(board)
        draw.text((10,10),'HAND DETAIL DIAGNOSTIC - same avatar/pose for neural vs raw 3D; no sharpness claim',fill='black')
        captions=['Full-view baseline enlarged','Focused neural render','Focused raw 3D (GS)']
        for row,(i,frames) in enumerate(zip(ids,pairs)):
            source=Image.fromarray(baseline[i,80:330,30:488]).resize((w,h),Image.Resampling.BICUBIC)
            for col,im in enumerate([source]+[Image.fromarray(f) for f in frames]):
                y=50+row*(h+32)
                board.paste(im,(col*w,y+24))
                draw.text((col*w+8,y),f'Frame {i}: '+captions[col],fill='black')
        board.save(output/'hand-renderer-comparison.png')
        report.update(frame_ids=ids,rendered_frames=6,cuda_compute_events=len(kernels),
            render_seconds=time.monotonic()-started,focused_shape=[h,w,3],
            reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            reference_size=list(Image.open(args.reference).size),
            before_close=json.loads(json.dumps(backend.stats)),
            same_cache_verified=True,
            limitations=['Raw GS appearance is diagnostic, not a production replacement',
                         'The focused crop excludes parts of the body',
                         'Enlarging the baseline adds no recovered detail',
                         'Baseline and focused views have separate reconstructions',
                         'Same-cache guarantee applies to focused neural versus GS only'])
        backend.close()
        report['after_close']=json.loads(json.dumps(backend.stats))
        stats=report['after_close']
        if stats['cache_present'] or not stats['model_python_released'] or stats['allocated_after_close']>1024**2:
            raise RuntimeError('Cleanup check failed')
        report['execution_passed']=True
    except Exception:
        report['error']=traceback.format_exc()
    finally:
        if backend is not None:backend.close()
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('HAND_RECONSTRUCTION_DIAGNOSTIC',json.dumps(report),flush=True)
    return 0 if report['execution_passed'] else 1

if __name__=='__main__':
    raise SystemExit(main())
