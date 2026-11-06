# Finger twist correction

The 15-second avatar test exposed distorted raised hands. The old finger mapping constructed a full coordinate frame from each segment direction and a projected palm-width hint. When a curled segment crosses the palm-width direction, that hint changes sign and the computed frame can acquire a 180-degree axial flip.

The revised mapping applies the shortest rotation between the parent-rotated rest bone and the measured segment. It preserves inherited axial twist because a segment direction cannot observe that twist. Exactly reversed/ambiguous segments are rejected before temporal state is committed. Missing observations ease toward neutral instead of instantly resetting; fully missing body packets still reset the trackers.

A regression crossing 80 to 100 degrees of curl fails against the old implementation and passes against the new one for both hands. New tests also cover smooth relaxation and rejection without corrupting temporal state. All 23 CPU tests and Python compilation passed.

This changes only opt-in head/hand refinement; it does not add facial expressions, lip sync or live streaming. It cannot reconstruct sharp fingers absent from the avatar representation. A lower joint-rotation magnitude is a diagnostic, not proof of anatomical accuracy.

## Actual 15-second evidence

The same private 180-frame, 12-FPS clip was processed on the existing Kaggle T4 session. Hand detection coverage stayed at 169 left / 173 right frames. The old packets contained 18 finger-joint rotation samples above 150 degrees; the new packets contained zero. Maximum left-finger local rotation fell from 3.1354 to 1.7656 radians. Body joints other than the two wrists, head calibration, camera fields and packet timing were identical. This comparison does not establish anatomical ground truth. The avatar is reconstructed once per run using that run's initial pose, so pixel differences are not a strictly identical-cache ablation.

The actual T4 render completed 180/180 frames with 170,100 CUDA compute events, no held frames and no foreground edge contacts. Setup took 42.311 seconds; the profiled render took 114.677 seconds. One model load and reconstruction were used, and cleanup released the cache/model, leaving 6,144 CUDA allocated bytes. This is offline performance, not live FPS.

The downloaded output was independently probed at 518x630, 180 frames, 12 FPS and 15.000 seconds. A labelled synchronized comparison preserves those 180 source moments and adds enlarged upper-body views; it does not synthesize sharper detail. Visual review still shows blurred/fused fingers and distorted hand silhouettes in some poses. The geometric regression is fixed, but the visual hand-quality requirement remains unmet. Further work needs to evaluate the avatar reconstruction's hand representation before adding live streaming or claiming lifelike output.

No paid service was started; master was not modified.
