# LHM++ video-driven mouth opening: first bounded check

The existing LHM++ worker already accepts jaw_pose and expr. The body/head/hand tracker had emitted neutral values. This stage adds opt-in jaw opening only, leaving the existing worker, body/head/hands packets, identity and default baseline unchanged.

MediaPipe jawOpen is mapped to a positive SMPL-X jaw X rotation bounded at 0.4 radians, with smoothing 0.65. Other facial expression coefficients are preserved. ARKit blendshape scores are not copied into the unrelated FLAME expression basis. Missing detections relax the jaw toward closed. This is neither audio-driven lip sync nor fine lip-shape reconstruction.

## Actual free Kaggle T4 execution

- Initial full-frame face pass: 0/180 detections. The existing body-landmark head crop improved this to 172/180 (95.56%). No face-recognition identity matching is claimed.
- Google Face Landmarker float16 v1 task: SHA256 64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff, 3,758,596 bytes. MediaPipe 0.10.35 ran on CPU in an isolated Python 3.10 environment. Source: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
- Observed jawOpen range: 0.00000836..0.03304159. Smoothed jaw rotation max 0.00885750 radians (~0.51 degrees). The existing hand/body-motion clip does not excite visible mouth motion enough to validate speech.
- Bounded renderer probe: fixed source body/head/hands frame 99, same fictional open-hand reference and framing box [430,130,1030,864], render height 630. Three stills compare neutral, tracked maximum and controlled 0.4-radian opening. A separate profiled frame recorded 945 CUDA compute events. Strict weights loaded; process exit 0. Setup plus probe took 70.694 seconds, not a live FPS benchmark.
- Controlled opening visibly changes the mouth, though face/lip reconstruction remains coarse. The tracked opening is almost indistinguishable from neutral. This verifies a jaw parameter response, not natural speech shapes, visual quality or synchronization.
- Backend close reported 6,144 allocated / 4,194,304 reserved bytes; external nvidia-smi after exit showed both GPUs at 0 MiB. Do not claim zero in-process allocation.
- CPU regression tests cover non-mutation/preservation of body, hands and expressions, bounded smoothing, missing-face relaxation, and invalid-score state safety; 2 passed.

The restored runtime's pretrained_models symlink had again become an empty file. The existing backup was preserved; a second uniquely named empty-file backup was made before reconnecting the manifest-verified prior folder. The original restoration guard correctly refused overwriting its first backup. No baseline Python source or desktop code was changed.

## Reproduce

Run scripts/mouth_tracking_test.py with --model, --model-sha256, --video, --motion-jsonl and a new --output-dir. It requires the complete existing 180-frame sequence and validates source timestamps. Run scripts/mouth_avatar_probe.py with prepared --runtime-root, --model-dir, --reference, the resulting --motion-jsonl, a new --output-dir and --framing-box 430 130 1030 864. The CUDA environment needs the existing lhmpp_* helper modules on the script path. Set MPLBACKEND=Agg for headless GPU subprocesses.

No new full video is presented as improved lip sync: the source lacks sufficient detected mouth movement. Next obtain a well-lit speaking/open-close-mouth clip, inspect tracked face motion, then compare the full LHM++ replay. Fine lip shaping, expression-basis mapping, audio alignment, live streaming and hand quality remain unfinished. No paid service was started; master is untouched.
