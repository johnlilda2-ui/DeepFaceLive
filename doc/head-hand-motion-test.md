# Neutral head calibration and hand detail — 6 November 2026

This opt-in stage visibly improves the head posture and raised-hand detail on the
existing consented three-second recording. It is **not a claim that hands are
fully reliable**. The unchanged body-only pipeline remains the default.

## Changes

- `avatar_head_hands.py` adds explicit neutral-head calibration. The measured
  local head rotation is right-composed with the inverse neutral feature frame.
  This removes the nose/mouth depth bias at a known level/front-facing reference,
  while preserving subsequent relative nod/turn rotation. Calibration is not
  silently applied by the original body adapter.
- The hand model runs on CPU with a pose-guided crop around each body wrist.
  Detections must spatially match that wrist; handedness class labels alone are
  not used to assign left/right. Nonfinite, implausible or degenerate geometry
  is rejected. Rest-joint geometry and fingertip positions are computed from the
  existing checksum-verified SMPL-X asset. Model files and rest coordinates are
  not redistributed here.
- Palm and finger segment frames produce the wrist and 15 local hand rotations
  per side, respecting the SMPL-X index/middle/pinky/ring/thumb order. Rotation
  smoothing reduces jitter. Missing/rejected detections retain the original body
  wrist and neutral fingers, and reset smoothing; no indefinite stale hand hold.
- The replay CLI gains optional `--render-height`, `--framing-box` and
  `--detail-kind head-hands`. Existing defaults remain 420-pixel, uncropped,
  body-only output. Detail labels report provenance; they do not grant consent.
- The tested tight camera is `430 130 1030 864`, height 630, producing native
  **518×630** pixels. The subject is about 540 pixels tall rather than roughly
  313 in the old wide render. This change re-renders at a tighter camera;
  the improvement is not only display upscaling.

## Actual test and audit

Semantic source: `4e2c4e316eed5897205e08e7fba03f2dc136f8f0`.
[Aggregate execution record](evidence/lhmpp-t4-head-hand-summary.json).
[Hand-model manifest](evidence/mediapipe-hand-landmarker-v1.json).

CPU tracking accepted 18 frames. Hand detections were observed in **14/18 left**
and **13/18 right** frames, exceeding the predeclared minimum of 12 per side.
Maximum observed finger-joint rotations were 1.315 and 1.262 radians. These are
kinematic observations, not independent finger-angle accuracy measurements.

The explicit neutral reference's local head-frame rotation magnitude was
**41.949° before calibration and 0° afterward**. This zero is a consequence of
the calibration definition, not an independent measurement of a physical
head-angle error. Later local head rotations reached 11.622°. Visual comparison
of the matched raised-arm frame shows the target looking forward rather than
downward.

The T4 replay rendered **18/18** frames with **17,010 CUDA compute events**, one
model load and one avatar reconstruction. No foreground contacted the frame
border. Model/cache release passed: no close error, 6,144 allocated bytes and
4 MiB reserved after cleanup. Setup was 58.865 seconds and profiled rendering
12.972 seconds; this is not a live FPS benchmark.

A post-fix limb-consistency screen passed on 18 matched frames / 144 bones:
mean 4.215°, p90 7.554°. It reuses the same detector and is not independent
ground truth for head/fingers. Downloaded output was separately verified as
H.264/yuv420p, 518×630, 18 frames, three seconds. Manual before/after comparison
shows a substantially more upright head and more distinct raised finger shapes;
the same fictional target and clothing remain.

**15 CPU regression tests pass**: the original eight body geometry cases and
seven new calibration/hand cases. They cover calibration-before-use, composed
relative nod/turn preservation, no mutation of input packets, rest hands,
global palm rotation without false finger curl, local parent-relative finger
bending on both sides, and rejected geometry not poisoning temporal state.
Python sources compile. The new prior-placeholder restore repair also completed
successfully on this actual Kaggle restart.

## Reproduction

After the existing body tracking test passes, run `head_hand_tracking_test.py`
in the isolated CPU environment. Supply the verified hand model, SMPL-X asset
and its original asset manifest, body report/JSONL, matching driving clip and
an explicitly known neutral frame (`--neutral-frame 0` in this test).
Its output directory must be new.

Replay the resulting JSONL with the prepared GPU environment using:
`--driver-kind consented-human --detail-kind head-hands --render-height 630
--framing-box 430 130 1030 864`, plus the existing runtime/model/reference,
driver-video and output-dir arguments. The original reference is the same
fictional AIGC person used in preceding stages.

The official [MediaPipe Hand Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
describes 21 hand landmarks; [SMPL-X joint order and tip mapping](https://github.com/aigc3d/LHM-plusplus/tree/906b5d9fb967ab42efb92f6fa55bf22cac86b653/core/models/rendering/smplx/smplx)
were inspected in the pinned renderer. This code computes its own segment-frame
mapping; it does not copy or distribute the model assets.

## Remaining limits

Hand detections are missing in four left and five right frames. Neutral fallback
can visibly jump, so hand continuity still needs work. Single-image reconstruction
also limits finger appearance; pose tracking cannot restore missing texture or
guarantee anatomical detail. Thumb/twist mapping remains an approximation.

Neutral calibration requires a known level pose. Turns, nods, occlusion and
recalibration need further real-input validation. The tight fixed camera is
verified only for this bounded motion and may clip wider movement. Jaw, eyes
and expressions remain neutral; no lip sync, live webcam transport or OBS
output was added. No paid service was started. Master, desktop application
and existing face worker are unchanged. Private media/landmarks are excluded
from the repository; result media was returned directly to the user.
