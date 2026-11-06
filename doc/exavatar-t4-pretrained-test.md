# ExAvatar pretrained T4 test

The official pretrained author-demo avatar was executed on the existing free Kaggle T4. This test did not train a target avatar, replace LHM++, modify the desktop application, or start paid compute.

## Observed result

| Render resolution | Timed frames | Synchronized render FPS | Median | P95 | Peak PyTorch allocated memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512x512 | 180 | 7.386 | 135.222 ms | 138.678 ms | 2,829,885,440 bytes |
| 1024x1024 | 30 | 7.466 | 133.572 ms | 137.556 ms | 2,839,259,136 bytes |

These unprofiled timings include human animation and Gaussian rendering, synchronized after each frame. They exclude GPU-to-CPU image transfer, video encoding, webcam tracking, transport and OBS. There were three warm-up renders per resolution. A separate single-frame profiler recorded **2,006 CUDA compute events**, so profiling overhead is excluded from the speed measurements.

The human weights loaded strictly. The official scene/background Gaussian model and optimizer state were not moved to the GPU. The process reported 43,713,536 CUDA allocated bytes before exit (imported global model helpers remain resident); a subsequent external nvidia-smi read showed **0 MiB used** after the process ended. Do not describe the in-process allocation as zero.

The downloaded offline H.264 preview was independently probed: 180 frames, 512x512, 12-FPS playback, 15.000 seconds, 135,314 bytes. The 12-FPS playback setting is not a measured live rate.

## Visual audit and decision

Reviewed frames 7, 24, 48, 90 and 150 show distorted/closed hand shapes. The adapter also places the head against or partly beyond the upper frame edge. Execution and speed are verified; visual hand quality, complete framing and retargeting compatibility are **not** accepted. This does not isolate whether the remaining hand errors come from the reused pose approximation, its mapping into ExAvatar, or the trained avatar. It is not evidence that ExAvatar itself can never produce accurate fingers.

**Do not replace the working avatar pipeline or spend paid GPU credit based on this test.** An engineering follow-up would first validate canonical open-hand/flexion poses and the camera projection convention, then compare output quality before considering personalized training. The current adapter does not yet meet the user's lifelike full-body live-video requirement.

## Sources and preparation

- ExAvatar official source: https://github.com/mks0601/ExAvatar_RELEASE at `518f5647c8fa75cb648315c1677fdc105101fe11`.
- Depth/mask rasterizer linked upstream: https://github.com/leo-frank/diff-gaussian-rasterization-depth at `03f0b7d00383d6e96c22b37325ac9e5450947bf5`.
- Official checkpoint and identity download links are in ExAvatar's `avatar/README.md`.
- Checkpoint download: 3,964,944,173 bytes. Identity ZIP: 3,581,008,471 bytes; only shape_param.json, face_offset.json, joint_offset.json and locator_offset.json were read using verified HTTP ranges, rather than downloading all training images/video.
- Runtime files live under `/kaggle/working/dfl-exavatar-probe`. An isolated `/tmp/dfl-exavatar-env` reads existing prepared GPU dependencies, without changing the working LHM++ environment.
- Rasterizer build for T4 compute capability 7.5 needed an explicit `#include <cstdint>` in `cuda_rasterizer/rasterizer_impl.h` on Kaggle CUDA 12.8. The build then passed.
- The inference overlay removes one verified-unused `from utils.flame import flame` import from the upstream nets module. It directly instantiates the unchanged HumanGaussian and GaussianRenderer used by the official animate.py path. Training-only face-texture helpers/losses and scene loading are omitted; geometry, colour networks, skinning and rasterization calculations are unchanged. Existing prepared SMPL-X and FLAME geometry assets are reused; missing FLAME_texture.npz is not fabricated.
- CPU memory mapping loads the checkpoint, and the human state dict is loaded with strict=True.
- Run `scripts/exavatar_pretrained_probe.py MOTION_JSONL` with the prepared isolated Python runtime and the existing complete 180-frame motion file.

The initial unused-import guard mistakenly counted both occurrences of the word flame in the import itself; it was corrected to check the rest of the module for actual additional references. Compilation passed after the correction, and the subsequent actual GPU run produced the results above.

Master remains unchanged at `fc7b787bda2b8c186e142c52857458eea3a935ed`. No private source footage, per-frame pose coordinates or model weights are included in this commit.

## Bounded follow-up and stopping decision

A final three-still diagnostic ran successfully on the free Kaggle Tesla T4 with strictly loaded official human weights. It compared frame 24 with tracked hands, the same body pose with both hand rotation vectors zeroed, and the official canonical body pose with zeroed hands. These diagnostics do not establish that zero rotations correspond to anatomically open hands.

The camera audit found two adapter problems: upstream get_proj_matrix ignores principal-point offsets, and HumanGaussian defaults to interpreting the supplied pose in camera coordinates. A view translation would be cancelled by its camera-to-world conversion. The bounded script therefore explicitly uses is_world_coord=True and centers with view translation. Rendered foreground margins, measured at 1024 square, passed a 40-pixel minimum:

| Case | Left | Top | Right | Bottom |
| --- | ---: | ---: | ---: | ---: |
| Tracked hands | 351 | 97 | 344 | 104 |
| Zero hand rotations | 350 | 97 | 360 | 104 |
| Canonical body and hands | 96 | 97 | 96 | 292 |

The first diagnostic attempted view translation without world-coordinate mode and correctly failed its framing gate (top margin zero); the audited correction above passed. Visual review of the corrected three-still comparison still shows curled/fused hand shapes. Framing is fixed for these three cases; hand quality is not accepted. No claim is made that the upstream model cannot reconstruct hands: this check has not isolated the pose convention, adapter and trained avatar.

**Stopping decision: stop this ExAvatar experiment and retain the existing LHM++ open-hands baseline.** The LHM++ worker and desktop application remain unchanged. No personalized ExAvatar training, replacement deployment, new branch or paid service was started. Future work should continue from the existing LHM++ baseline; its hand quality, lip sync and live streaming remain unfinished.

Reproduce with scripts/exavatar_bounded_hand_check.py MOTION_JSONL in the isolated Python 3.10 environment. A Kaggle restart had cleared /tmp, so the minimal torch 2.3.0/cu121, torchvision 0.18.0, pytorch3d 0.7.6, numpy 1.23.0, scipy 1.13.1, smplx 0.1.28 and chumpy 0.70 dependencies and manifest-verified SMPL-X/FLAME geometry plus static/dynamic landmark embeddings were restored. The retained compiled rasterizer and checkpoint were reused. No LHM++ environment was edited.
