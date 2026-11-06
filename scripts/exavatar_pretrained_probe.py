"""Bounded ExAvatar pretrained human-only inference and unprofiled T4 probe.
Uses the official animation path; excludes scene and training-only losses.
"""
import os,sys,json,time,traceback,hashlib,copy,gc,subprocess,shutil
from pathlib import Path
import numpy as np

ROOT=Path('/kaggle/working/dfl-exavatar-probe')
PIN='518f5647c8fa75cb648315c1677fdc105101fe11'
report={'scope':'official_exavatar_pretrained_human_inference','passed':False,
        'upstream_commit':PIN,'target':'official author demonstration avatar',
        'live_verified':False,'quality_verified':False}
def main():
    human=renderer=checkpoint=state=None
    try:
        runtime=ROOT/'runtime'
        if subprocess.check_output(['git','-C',str(runtime),'rev-parse','HEAD'],text=True).strip()!=PIN:
            raise RuntimeError('Unexpected ExAvatar revision')
        overlay=ROOT/'inference-overlay';(overlay/'nets').mkdir(parents=True,exist_ok=True)
        source=(runtime/'avatar/common/nets/module.py').read_text()
        unused='from utils.flame import flame'
        if source.count(unused)!=1 or source.count('flame')!=1:
            raise RuntimeError('The unused FLAME import needs review')
        (overlay/'nets/module.py').write_text(source.replace(unused,'# Unused training-only FLAME texture helper excluded'))
        shutil.copy2(runtime/'avatar/common/nets/layer.py',overlay/'nets/layer.py')
        (overlay/'nets/__init__.py').write_text('')
        sys.path[:0]=[str(overlay),str(runtime/'avatar/main'),str(runtime/'avatar/common')]
        import torch
        from PIL import Image,ImageDraw
        import imageio.v3 as iio
        from config import cfg
        cfg.human_model_path='/tmp/dfl-fullbody-assets/prior/human_model_files'
        cfg.dataset='Custom'
        from utils.smpl_x import smpl_x
        from nets.module import HumanGaussian,GaussianRenderer
        report.update(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
                      gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
        smpl_x.set_id_info(*[torch.tensor(json.loads((ROOT/name).read_text()),dtype=torch.float32)
            for name in ['shape_param.json','face_offset.json','joint_offset.json','locator_offset.json']])
        checkpoint=torch.load(ROOT/'snapshot_4.pth',map_location='cpu',mmap=True,weights_only=False)
        state={k[len('human_gaussian.'):]:v for k,v in checkpoint['network'].items()
               if k.startswith('human_gaussian.')}
        if not state:raise RuntimeError('No human weights in the official checkpoint')
        human=HumanGaussian().cuda()
        with torch.no_grad():human.init()
        human.load_state_dict(state,strict=True)
        human.eval();renderer=GaussianRenderer().cuda()
        del state,checkpoint;state=checkpoint=None;gc.collect()
        report.update(strict_human_weights_loaded=True,training_texture_required=False,
            adaptation='Removed one unused import; instantiated official HumanGaussian and GaussianRenderer directly')
        motion_path=Path(sys.argv[1])
        records=[json.loads(s) for s in motion_path.read_text().splitlines() if s.strip()]
        if len(records)!=180 or [r['frame_id'] for r in records]!=list(range(180)):
            raise ValueError('Require the existing complete 180-frame motion sequence')
        def pose(i):
            raw=records[i]['pose']
            result={k:torch.tensor(raw[k],dtype=torch.float32,device='cuda').flatten()
                    for k in ['body_pose','jaw_pose','leye_pose','reye_pose','lhand_pose','rhand_pose']}
            result.update(root_pose=torch.tensor([np.pi,0.,0.],device='cuda'),
                          trans=torch.tensor([0.,0.,4.],device='cuda'),
                          expr=torch.zeros(smpl_x.expr_param_dim,device='cuda'))
            return result
        def camera(size,fx=1.,cx=0.,cy=0.):
            return dict(R=torch.eye(3,device='cuda'),t=torch.zeros(3,device='cuda'),
                        focal=torch.tensor([fx*size,fx*size],dtype=torch.float32,device='cuda'),
                        princpt=torch.tensor([size*(.5-cx*fx),size*(.5-cy*fx)],dtype=torch.float32,device='cuda'))
        # Compute one fixed camera from all selected raised/lowered-hand bounds.
        extents=[]
        with torch.inference_mode():
            for i in [0,7,24,48,90,150,179]:
                assets,*_=human(pose(i),camera(512))
                xyz=assets['mean_3d'];uv=xyz[:,:2]/xyz[:,2:]
                if not torch.isfinite(uv).all() or xyz[:,2].min()<=0:
                    raise RuntimeError('Invalid camera-space geometry')
                extents.append(torch.stack([uv.min(0).values,uv.max(0).values]).cpu().numpy())
            bounds=np.stack(extents);lo=bounds[:,0].min(0);hi=bounds[:,1].max(0)
            center=(lo+hi)/2;f=.82/max(hi-lo)
        report['fixed_camera']={'normalized_focal':float(f),'center':center.tolist()}
        def frame(i,size):
            with torch.inference_mode():
                assets,*_=human(pose(i),camera(size,f,*center))
                return renderer(assets,(size,size),camera(size,f,*center))['img']
        timings={};saved=[]
        for size,ids in [(512,list(range(180))),(1024,[0,7,24,48,90,150]*5)]:
            for _ in range(3):frame(24,size)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            elapsed=[];frames=[]
            for n,i in enumerate(ids):
                start=time.perf_counter();rgb=frame(i,size);torch.cuda.synchronize()
                elapsed.append(time.perf_counter()-start)
                if not torch.isfinite(rgb).all():raise RuntimeError('Nonfinite rendered pixels')
                pixels=rgb.clamp(0,1).mul(255).byte().permute(1,2,0).cpu().numpy()
                if np.mean(pixels<245)<.01:raise RuntimeError('Blank rendered frame')
                if size==512:frames.append(pixels)
                elif n<6:
                    Image.fromarray(pixels).save(ROOT/f'probe-1024-{i}.png')
                if n%30==0:print('EXAVATAR_FRAME',size,n,flush=True)
            timings[str(size)]={'frames':len(ids),'fps':len(ids)/sum(elapsed),
                'median_ms':float(np.median(elapsed)*1000),'p95_ms':float(np.percentile(elapsed,95)*1000),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'scope':'GPU-synchronized human animation and render; excludes transfers, encoding, tracking and network'}
            if size==512:iio.imwrite(ROOT/'exavatar-demo-15s.mp4',np.stack(frames),fps=12,
                                    codec='libx264',pixelformat='yuv420p',macro_block_size=2)
            del frames
        # One separate profiled frame proves CUDA execution without distorting the speed run.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            frame(24,512);torch.cuda.synchronize()
        events=[e for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA
                and not e.name.startswith(('Memcpy','Memset'))]
        if not events:raise RuntimeError('No CUDA compute evidence')
        report.update(passed=True,cuda_compute_events=len(events),timings=timings,
                      output_frames=180,output_fps=12,output_duration_seconds=15,
                      limitations=['Official demonstration identity, not the generated target',
                        'No background, tracking, network, lip sync or OBS in benchmark',
                        'Inherited body/hand pose approximation; no anatomical accuracy claim',
                        'Preview playback speed does not establish live performance'])
    except Exception:
        report['error']=traceback.format_exc()
    finally:
        human=renderer=state=checkpoint=None;gc.collect()
        if 'torch' in locals():
            torch.cuda.empty_cache()
            report['allocated_after_cleanup']=torch.cuda.memory_allocated()
        (ROOT/'exavatar-probe-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('EXAVATAR_PROBE_REPORT',json.dumps(report),flush=True)
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
