"""Private stdio GPU renderer child. No socket, camera or public listener."""
import contextlib
import json
import sys
import traceback
from pathlib import Path


def main():
    output=sys.stdout.buffer
    worker=None
    def header(value):
        output.write((json.dumps(value,allow_nan=False)+'\n').encode());output.flush()
    try:
        config=json.loads(Path(sys.argv[1]).read_text())
        # Upstream model initialization logs must never corrupt the frame pipe.
        with contextlib.redirect_stdout(sys.stderr):
            from headless_avatar_worker import HeadlessAvatarWorker
            if config.get('backend') == 'splattingavatar':
                from splatting_avatar_backend import SplattingAvatarBackend
                backend = SplattingAvatarBackend(config)
            else:
                from lhmpp_avatar_backend import LHMAvatarBackend
                backend = LHMAvatarBackend(config['runtime_root'],config['model_dir'],config['reference'],
                    render_height=config.get('render_height',630),
                    framing_box=config.get('framing_box'),optimized_decoder=True)
            worker=HeadlessAvatarWorker(backend,config['initial_pose'])
        header(dict(type='ready',shape=list(backend.output_shape),gpu=backend.stats['gpu']))
        profiled=False
        while True:
            line=sys.stdin.buffer.readline(16385)
            if not line:break
            if len(line)>16384 or not line.endswith(b'\n'):
                raise ValueError('Oversized render request')
            request=json.loads(line)
            if request=={'type':'close'}:break
            if set(request)!={'type','id','timestamp_ms','pose'} or request['type']!='render':
                raise ValueError('Invalid render request')
            with contextlib.redirect_stdout(sys.stderr):
                kernels=0
                if config.get('profile_first_frame',False) and not profiled:
                    import torch
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
                        rgb=worker.render(request['id'],request['timestamp_ms'],request['pose'])
                        torch.cuda.synchronize()
                    kernels=sum(e.device_type==torch.autograd.DeviceType.CUDA and
                                not e.name.startswith(('Memcpy','Memset')) for e in profile.events())
                    if not kernels:raise RuntimeError('No actual CUDA compute events')
                    profiled=True
                else:rgb=worker.render(request['id'],request['timestamp_ms'],request['pose'])
            pixels=rgb.tobytes()
            header(dict(type='frame',id=request['id'],bytes=len(pixels),cuda_compute_events=kernels))
            output.write(pixels);output.flush()
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        header(dict(type='error',message='GPU avatar renderer failed; inspect the private worker log'))
        return 1
    finally:
        if worker is not None:
            with contextlib.redirect_stdout(sys.stderr):worker.close()


if __name__=='__main__':raise SystemExit(main())
