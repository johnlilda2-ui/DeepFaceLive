# Enlarged body-motion comparison

This stage makes the existing recorded test easier to judge. It does not run new
inference or change the avatar's pose, geometry, face, hands or background.

## Review layout

`scripts/avatar_motion_comparison.py` decodes the source recording and its matching
avatar render to the declared review timeline. It displays them side by side with
plain labels. The avatar uses one fixed crop covering the union of nonwhite pixels
across all frames, plus a margin. This avoids camera pumping and retains the whole
visible figure. The threshold assumes the existing white renderer canvas; it is
not semantic segmentation or an alpha mask.

The native 742×420 avatar frames are preserved during decoding. An audit corrected
an unnecessary downsample before enlargement. Source recording display is resized
to a 480-pixel width. Both panels preserve aspect ratio. Enlargement does not add
detail. The white renderer background is retained truthfully.

Example using the previously tested three-second window:

```bash
python scripts/avatar_motion_comparison.py --driver-video /absolute/source.mp4 --avatar-video /absolute/avatar.mp4 --driver-start 10 --output-dir /absolute/new-review
```

The output directory must not exist. Default review is 18 unique frames at 6 FPS,
replayed three times. The nine-second review contains 54 encoded frames, not nine
seconds of newly generated movement.

## Actual verification — 6 November 2026

- Source window: seconds 10–13 of the consented recording, matching the previous
  tracking test. No additional external upload or GPU session was needed.
- H.264/yuv420p output: 1080×800, 6 FPS, 54 frames, nine seconds, no audio.
- Fixed native-avatar crop: (247, 82) to (429, 420), within the 742×420 input.
- Avatar display tile: 293×545. Visible body height is approximately 500 pixels,
  versus approximately 313 pixels in the previous raw render. This is display
  enlargement, not higher-resolution inference.
- Python compilation passed. ffprobe independently confirmed output geometry,
  frame rate, frame count and duration. First, middle and last unique poses were
  inspected after encoding; head and feet remain visible, arm raises correspond,
  and the fixed crop remains stable.
- The comparison video and preview were returned privately to the user. Personal
  source footage and the side-by-side media are excluded from this repository.

## Head posture audit

Current head rotation uses an ear-to-ear axis and a short nose-minus-mouth-midpoint
vector as the head-up hint in `avatar_body_retarget.py`. That vector contains
anatomical depth as well as vertical displacement. A direct mathematical check
using equal 3 cm vertical/depth components produces a 45-degree pitch even with a
level chest. This demonstrates a possible systematic bias; it is not a measured
head-angle error for this user.

The enlarged comparison visibly shows the avatar looking downward while the
driver faces forward. Head posture needs a separate calibration/estimation test
with neutral pose and known turns/nods before changing production pose mapping.
No speculative correction was applied here. Hands/fingers and lip sync are also
unfinished. The earlier limb-direction score does not validate these features.

The desktop application, face worker and master remain untouched. This utility
is a review/export stage, not live webcam transport, background replacement or OBS.
