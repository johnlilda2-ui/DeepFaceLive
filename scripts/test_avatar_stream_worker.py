import unittest
import numpy as np
from headless_lhmpp_stream_worker import HeadlessLHMStreamWorker


class Tracker:
    def __init__(self):self.missing=False;self.times=[];self.resets=0
    def reset(self):self.resets+=1
    def convert(self,image,timestamp):
        self.times.append(timestamp)
        return (None if self.missing else {}),dict(body=not self.missing)
    def close(self):pass


class Renderer:
    shape=[14,14,3]
    def __init__(self):self.calls=[];self.closed=False
    def render(self,frame_id,timestamp,pose):
        self.calls.append((frame_id,timestamp))
        result=np.zeros(self.shape,dtype=np.uint8);result[:,:,0]=123
        return result
    def close(self):self.closed=True


class StreamWorkerTests(unittest.TestCase):
    def test_two_sessions_reset_media_clock_keep_gpu_order_and_rgb_conversion(self):
        tracker,renderer=Tracker(),Renderer();worker=HeadlessLHMStreamWorker(tracker,renderer)
        frame=np.zeros((14,14,3),dtype=np.uint8)
        a,status=worker.process_video(frame,1000.)
        worker.process_video(frame,1100.)
        worker.reset_motion()
        worker.process_video(frame,0.)
        worker.process_video(frame,100.)
        self.assertEqual(tracker.times,[0.,100.,0.,100.])
        self.assertEqual([i for i,t in renderer.calls],[0,1,2,3])
        self.assertTrue(all(a[1]>b[1] for a,b in zip(renderer.calls[1:],renderer.calls)))
        self.assertEqual(status,'avatar');self.assertEqual(a[0,0].tolist(),[0,0,123])
        worker.close();self.assertTrue(renderer.closed)

    def test_tracking_loss_cannot_return_source_or_stale_avatar(self):
        tracker,renderer=Tracker(),Renderer();worker=HeadlessLHMStreamWorker(tracker,renderer)
        tracker.missing=True
        out,status=worker.process_video(np.full((14,14,3),255,dtype=np.uint8),0.)
        self.assertEqual(status,'avatar_no_body');self.assertFalse(out.any())
        self.assertEqual(renderer.calls,[])
        with self.assertRaises(ValueError):worker.process_video(out,0.)
        worker.close()


if __name__=='__main__':unittest.main()
