"""Actual portrait GPU motion and appearance test using fictional project samples."""
import argparse
import json
from pathlib import Path
import tempfile
import time


def main():
    import cv2
    import numpy as np
    import torch
    from headless_portrait_worker import HeadlessPortraitWorker
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-dir',required=True)
    parser.add_argument('--source-image',required=True)
    parser.add_argument('--driver-image',required=True)
    parser.add_argument('--second-driver-image',required=True)
    parser.add_argument('--output-dir',required=True)
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True,exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='run-',dir=root))
    worker = HeadlessPortraitWorker(args.source_image,run,args.runtime_dir)
    first, second = cv2.imread(args.driver_image), cv2.imread(args.second_driver_image)
    assert first is not None and second is not None
    times, frames = [], []
    try:
        for index, driver in enumerate([first]*3+[second]*3):
            started = time.perf_counter()
            frame, status = worker.process_frame(driver)
            torch.cuda.synchronize()
            times.append((time.perf_counter()-started)*1000)
            assert status == 'avatar' and frame.shape == worker.source.shape
            assert cv2.imwrite(str(run/f'avatar-{index}.png'),frame)
            frames.append(frame)
        motion_change = float(np.abs(frames[-1].astype(float)-frames[0]).mean())
        assert motion_change > .1,'Different driver poses/expressions did not animate the portrait'
        held, status = worker.process_frame(np.zeros_like(first))
        assert status == 'avatar_no_face' and np.array_equal(held,frames[-1]), 'Missing face exposed the driver canvas'
        worker.reset_motion()
        reset, _ = worker.process_frame(first)
        assert float(np.abs(reset.astype(float)-frames[0]).mean()) < .1,'New session failed to reset motion calibration'
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
            worker.process_frame(second)
            torch.cuda.synchronize()
        cuda_events = sum(str(event.device_type).endswith('CUDA') for event in profile.events())
        profile.export_chrome_trace(str(run/'torch-cuda-trace.json'))
        assert cuda_events > 0,'No actual CUDA activity was captured for portrait rendering'
        h,w = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(run/'portrait-diagnostic.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),5,(w*2,h))
        assert writer.isOpened()
        for frame in frames:
            writer.write(np.concatenate((worker.source,frame),axis=1))
        writer.release()
    finally:
        worker.close()
    report = {'status':'passed','scope':'two fictional driving stills; not live webcam or full-body motion',
              'output_shape':list(frames[0].shape),'motion_change_mae':motion_change,
              'no_face_kept_portrait':True,'motion_reset_verified':True,'torch_cuda_events':cuda_events,
              'latency_ms':times,'warm_p50_ms':float(np.median(times[1:])),
              'execution_evidence':str(worker.profile_dir/'portrait-evidence.json')}
    (run/'portrait-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    print('PORTRAIT_GPU_TEST_PASSED',run,flush=True)


if __name__ == '__main__':
    main()
