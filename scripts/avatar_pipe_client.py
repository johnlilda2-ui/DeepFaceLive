"""Bounded Linux-cloud stdio transport between CPU tracking and GPU rendering."""
import json
import math
import os
from pathlib import Path
import select
import subprocess
import threading
import time
import numpy as np


class AvatarPipeClient:
    def __init__(self, gpu_python, config, log_path, startup_timeout=180., render_timeout=30.):
        if not math.isfinite(render_timeout) or not 0<render_timeout<=120:
            raise ValueError('Invalid render timeout')
        self.lock=threading.Lock();self.closed=False;self.failed=False
        self.render_timeout=render_timeout
        self.last_id=-1;self.last_timestamp_ms=-1.;self.cuda_compute_events=0
        self.log=Path(log_path).open('ab')
        try:
            self.process=subprocess.Popen([str(gpu_python),str(Path(__file__).with_name('avatar_render_pipe.py')),str(config)],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,bufsize=0,
                env={**os.environ,'MPLBACKEND':'Agg','CUDA_VISIBLE_DEVICES':'0'})
            ready=self._header(time.monotonic()+startup_timeout)
            self.shape=ready.get('shape')
            if (ready.get('type')!='ready' or not isinstance(self.shape,list) or len(self.shape)!=3 or
                    any(type(n) is not int for n in self.shape) or self.shape[2]!=3 or
                    not 14<=min(self.shape[:2])<=max(self.shape[:2])<=840):
                raise RuntimeError('Invalid GPU renderer startup response')
            self.gpu=ready.get('gpu')
        except BaseException:
            self.failed=True;self.close();raise

    def _read(self, count, deadline):
        result=bytearray()
        while len(result)<count:
            remaining=deadline-time.monotonic()
            if remaining<=0 or not select.select([self.process.stdout],[],[],remaining)[0]:
                raise TimeoutError('GPU frame pipe timed out')
            block=os.read(self.process.stdout.fileno(),min(count-len(result),65536))
            if not block:raise RuntimeError('GPU frame pipe closed')
            result.extend(block)
        return bytes(result)

    def _header(self, deadline):
        result=bytearray()
        while len(result)<8192:
            part=self._read(1,deadline)
            if part==b'\n':
                value=json.loads(result)
                if not isinstance(value,dict):raise RuntimeError('Invalid frame header')
                return value
            result.extend(part)
        raise RuntimeError('GPU frame header too large')

    def render(self, frame_id, timestamp_ms, pose):
        request=(json.dumps(dict(type='render',id=frame_id,timestamp_ms=timestamp_ms,pose=pose),
                            allow_nan=False,separators=(',',':'))+'\n').encode()
        if len(request)>16384:raise ValueError('Render request exceeds motion bound')
        if not self.lock.acquire(blocking=False):raise RuntimeError('GPU renderer is busy')
        try:
            if self.closed or self.failed:raise RuntimeError('GPU renderer is closed')
            self.process.stdin.write(request);self.process.stdin.flush()
            deadline=time.monotonic()+self.render_timeout
            header=self._header(deadline)
            count=int(np.prod(self.shape))
            if header.get('type')!='frame' or header.get('id')!=frame_id or header.get('bytes')!=count:
                raise RuntimeError('GPU renderer rejected frame or returned invalid dimensions')
            rgb=np.frombuffer(self._read(count,deadline),dtype=np.uint8).reshape(self.shape).copy()
            self.last_id,self.last_timestamp_ms=frame_id,timestamp_ms
            self.cuda_compute_events+=header.get('cuda_compute_events',0)
            return rgb
        except BaseException:
            self.failed=True
            if self.process.poll() is None:self.process.kill()
            raise
        finally:self.lock.release()

    def close(self):
        with self.lock:
            if self.closed:return
            self.closed=True
            process=getattr(self,'process',None)
            if process is not None:
                try:
                    if process.poll() is None:
                        process.stdin.write(b'{"type":"close"}\n');process.stdin.flush()
                        process.wait(timeout=5)
                except (OSError,subprocess.TimeoutExpired):
                    process.kill();process.wait(timeout=5)
                finally:
                    process.stdin.close();process.stdout.close()
            self.log.close()
