# Fifteen-second recorded motion test

This stage extends the existing opt-in avatar experiment on `experiment/cloud-headless-webrtc`. It does not change the desktop application or original face-swapping processor.

The private, consented driving excerpt contains 180 distinct frames at 12 FPS (15 seconds). This is new driving motion, not a repeated three-second animation. The target is the existing fictional AI reference.

## Tracking evidence

- CPU body tracking accepted 180/180 frames.
- CPU hand tracking observed the left hand on 169/180 frames and the right on 173/180.
- The explicit frame-zero head calibration removed a 43.229-degree initial local rotation; later measured local head rotations reached 19.697 degrees.
- Missing hands reset to the existing neutral-hand fallback, so tracking transitions can still pop.
- These counts measure detector coverage, not correct finger appearance or anatomically accurate tracking.
- Twenty local CPU geometry/timeline tests passed. Syntax checks passed.
- Bounds remain limited to 180 frames; default 18-frame tests remain unchanged. The 80% body coverage gate is unchanged. Sparse replay explicitly labels held frames.

## Reproduction

Use the already prepared isolated Kaggle CPU and GPU environments and pinned assets. Download all backend companions, including `lhmpp_t4_attention.py`, `lhmpp_decoder_precision.py`, and `lhmpp_camera_framing.py`; omitting these from a fresh script directory causes import failure.

Run `body_tracking_smoke_test.py --frames 180`, then `head_hand_tracking_test.py --frames 180 --neutral-frame 0`, then `body_tracking_avatar_replay.py --frames 180 --fps 12 --driver-kind consented-human --detail-kind head-hands --render-height 630 --framing-box 430 130 1030 864`, supplying the existing required asset, report, input and output paths. The neutral frame must be selected from a known level/front-facing frame rather than assumed for arbitrary uploads.

## Limits

This is an offline reconstructed avatar test. It is not proof of live streaming, lip sync, photorealistic appearance, correct finger geometry, or complete target-body identity transfer. Pelvis translation stays locked. Facial expressions are neutral. Private footage and per-frame tracking coordinates are not included in this repository.

## Actual GPU result

The Tesla T4 rendered all 180 frames with 170,100 recorded CUDA compute events. There were no held timeline frames or foreground edge contacts. The downloaded H.264 file was independently probed: 518 by 630, 12 FPS, 180 frames, 15.000 seconds, 286,172 bytes. One model load and one avatar reconstruction were used. Setup took 42.551 seconds and the profiled render took 116.382 seconds; this is not real-time throughput. Cleanup released the model and cache, leaving 6,144 CUDA allocated bytes.

Visual review at five points across the clip shows a more upright head, but distorted raised hands and unclear individual fingers. Passing execution and coverage checks does not mean the hand quality requirement is met. The next stage is hand-quality investigation, followed by a new recorded-motion review before live streaming.

No paid service was started. The master branch remains at fc7b787bda2b8c186e142c52857458eea3a935ed.
