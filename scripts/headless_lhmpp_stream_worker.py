"""Experimental serial avatar stream; CPU tracking and private GPU pipe."""
import math
import threading
import numpy as np


class HeadlessLHMStreamWorker:
    mode='avatar'
    engine='lhmpp'
    def __init__(self,tracker,renderer):
        self.tracker,self.renderer=tracker,renderer
        self.lock=threading.Lock()
        self.closed=False
        self.frame_id=getattr(renderer,'last_id',-1)+1
        self.last_render_timestamp=getattr(renderer,'last_timestamp_ms',-1.)
        self.media_origin=None
        self.last_media_timestamp=-1.
        self.stats=dict(rendered=0,tracking_missing=0,last_tracking={},live_quality_verified=False)

    def reset_motion(self):
        with self.lock:
            if self.closed:raise RuntimeError('Avatar stream is closed')
            self.tracker.reset()
            self.media_origin=None;self.last_media_timestamp=-1.
            self.previous_relative=0.

    def set_identity(self,image):
        raise ValueError('This avatar reference is fixed. Restart the avatar worker to change it.')

    def process_video(self,image,media_timestamp_ms):
        if (type(media_timestamp_ms) not in (float,int) or not math.isfinite(media_timestamp_ms) or
                media_timestamp_ms<0):raise ValueError('Invalid source media time')
        if (not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or
                image.shape[2]!=3 or min(image.shape[:2])<14 or max(image.shape[:2])>1920):
            raise ValueError('Invalid BGR input frame')
        if not self.lock.acquire(blocking=False):raise RuntimeError('Avatar stream is busy')
        try:
            if self.closed:raise RuntimeError('Avatar stream is closed')
            if media_timestamp_ms<=self.last_media_timestamp:raise ValueError('Stale source media frame')
            if self.media_origin is None:self.media_origin=media_timestamp_ms
            relative=media_timestamp_ms-self.media_origin
            packet,detail=self.tracker.convert(image,relative)
            self.last_media_timestamp=media_timestamp_ms
            self.stats['last_tracking']=detail
            if packet is None:
                self.stats['tracking_missing']+=1
                # Never leak the source face/body or silently reuse a stale avatar.
                return np.zeros(self.renderer.shape,dtype=np.uint8),'avatar_no_body'
            # GPU worker IDs stay monotonic across fresh WebRTC sessions; RTP
            # clocks reset per session, so use a separate monotonic render clock.
            timestamp=self.last_render_timestamp+max(1.,relative-getattr(self,'previous_relative',relative))
            rgb=self.renderer.render(self.frame_id,timestamp,packet)
            if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or tuple(rgb.shape)!=tuple(self.renderer.shape):
                raise RuntimeError('GPU renderer returned an invalid avatar frame')
            self.previous_relative=relative
            self.last_render_timestamp=timestamp;self.frame_id+=1
            self.stats['rendered']+=1
            return rgb[:,:,::-1].copy(),'avatar'
        finally:self.lock.release()

    def close(self):
        with self.lock:
            if self.closed:return
            self.closed=True
            try:self.tracker.close()
            finally:self.renderer.close()
