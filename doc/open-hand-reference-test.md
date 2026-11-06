# Open-hand fictional reference test

The user supplied a generated fictional full-body reference with open palms and separated fingers. This stage used the supplied upload as-is, checked its SHA-256 in Kaggle, and attached it as a private dataset to the existing notebook. It did not retrack or reupload the driving video.

Reference: 1024 by 1536 pixels; SHA-256 `fff33f6518ebc506238d3ea321726a8780448bfc83f4876a86dbe3756fff6eeb`.

The existing validated 180-frame head/hand motion packets were reused. Their timeline, calibration and motion content were unchanged. The new reference replaces the fictional bundled demonstration figure for this test only. No production default, original face-swap worker or desktop application was modified.

## Focused reconstruction probe

The bounded diagnostic rendered driving poses 7, 24 and 48 with neural and raw-GS output from the same reconstructed avatar cache. Six actual T4 renders completed, with 4,671 recorded CUDA compute events, one model load and one reconstruction. Finite-output and cleanup checks passed, leaving 6,144 CUDA allocated bytes.

The closer neural preview shows distinguishable finger contours in some poses. Distortion remains, so this is not a hand-quality acceptance pass. This new reference also changes identity, clothing and image resolution; it is not a controlled same-person experiment isolating finger visibility.

The diagnostic's full-view baseline input was the previous reference and was not used as the new reference's baseline or delivered as a matching comparison. The same-cache guarantee applies only to the new reference's focused neural-versus-GS pair.


## Full recorded-motion result

The same 180 packets were rendered at 12-FPS playback using the new reference. All 180 actual GPU renders passed, with 170,100 CUDA compute events, no held frames and no detected foreground edge contacts. One model load and one reconstruction were used. Setup took 42.457 seconds; the profiled render took 115.128 seconds. Cleanup released the model/cache, leaving 6,144 CUDA allocated bytes.

Independent probing of the downloaded H.264 output confirms 518x630, 180 frames, 12 FPS, 15.000 seconds and 266,357 bytes. Five selected full-body frames (7, 24, 48, 90, 150) were visually reviewed. The generated man's clothing and full body are present, and some finger contours are more distinguishable. Some hand shapes remain distorted; neither anatomical hand accuracy nor every pose's visual quality has been verified.

This is an offline motion-driven avatar result, not a live or lip-synced video. No paid services were started. Master remains at `fc7b787bda2b8c186e142c52857458eea3a935ed`.
