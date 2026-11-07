# Full-body avatar: lip, transport, OBS and hand stages

Development stays on `experiment/cloud-headless-webrtc`. The existing LHM++ offline baseline and DeepFaceLive desktop application remain intact. The face-swap server still defaults to face mode. Full-body streaming is an explicit `--mode avatar` option.

## What this implementation does

| Stage | Implemented | Remaining acceptance test |
| --- | --- | --- |
| Lips and audio timing | Private ARKit-to-FLAME lip-basis fitting, residual gate, bounded/time-smoothed coefficients; opt-in offline tracking and streaming; duration-matched audio export | Clear speaking clip, visual lip fidelity and speech synchronization. The live preview still transports video only. |
| Streaming and reconnection | CPU body/head/hands/face tracker, isolated persistent GPU renderer process, single pending video frame, media timestamps, fresh session clocks, reconnect control | User's browser/hotspot and TURN route, longer operation and measured end-to-end latency |
| OBS Virtual Camera | Dedicated processed-output window and disconnect clearing; no connection key in its URL | Actual Windows OBS Window Capture and Virtual Camera device test |
| Hands | Optional palm-normalized temporal observation guard; rejects abrupt bone-direction/length changes and relaxes missing/rejected hands | Sharp finger reconstruction remains unresolved; this filter does not create detail or fix fused Gaussian fingers |

## Lip mapping

`fit_avatar_lip_expression.py` reads only manifest-verified prepared FLAME and ARKit assets. It fits 100 FLAME coefficients to each supported lip displacement using ridge least squares. Relative reconstruction error above 0.45 rejects a control and zeros its mapping. No arbitrary coefficient indices are assigned. The derived NPZ stays private next to the prepared assets; it is not committed or distributed with the repository.

The actual fit accepted six controls: pucker, funnel, left/right press, upper roll and upper-up-left. It rejected left/right smile, lower roll, upper-up-right and left/right lower-down. This is an approximate geometric fit, not a validated expression model. Coefficient-vector norm is capped at six and time smoothing uses a 60 ms constant. The real source clip still lacks substantial measured mouth opening. Controlled stills show small changes around the mouth; low facial resolution remains a limitation.

`mouth_tracking_test.py --lip-map PRIVATE_MAP.npz` enables the mapping for offline packets; omit it to retain the earlier jaw-only behavior. Body/head/hands/camera/identity fields are preserved. A missing face relaxes lip coefficients rather than retaining a frozen expression. Neither a good fit nor changed rendered pixels establishes natural speech synchronization.

`avatar_audio_export.py --rendered REPLAY.mp4 --source ORIGINAL.mp4 --output NEW.mp4` muxes source audio without changing video speed. It accounts for the source audio/video start-time difference, rejects missing audio or mismatched duration, preserves inputs and refuses existing output paths. Optional `--audio-offset-ms` is bounded to two seconds. A real FFmpeg fixture with a delayed tone and a video flash tests the resulting A/V timing within 30 ms. This is an offline timeline test, not a live microphone or phoneme test.

## Prepared cloud configuration

Use the existing isolated LHM++ Python 3.10 environment for GPU rendering, and the separate MediaPipe CPU environment for signaling/tracking. Do not upgrade NumPy inside the prepared GPU environment to install MediaPipe.

The private JSON passed as `--avatar-config` contains:

- `runtime_root`, `model_dir`, `reference`, `initial_pose`, `render_height`, `framing_box` for the pinned LHM++ backend.
- `pose_model`, `hand_model`, `face_model`: the existing pinned Google MediaPipe task files, verified by SHA256 on startup.
- `smplx_model`, `manifest`: the existing prepared neutral SMPL-X geometry and asset manifest.
- Optional `lip_map`, `guard_hands` (default false), `calibrate_head_on_start` (default false), `profile_first_frame` (diagnostic only).

An operator who explicitly enables head calibration must begin with a level, front-facing neutral head. It is not automatically recalibrated every time body tracking disappears. The target reference is fixed for the GPU process. To use a different consented full-body reference, stop and restart that worker; the face-photo upload controls are hidden for avatar mode.

Start the prepared server with the CPU environment:

```bash
/tmp/dfl-face-env/bin/python /kaggle/working/dfl-pipeline-stages/headless_webrtc_server.py \
  --mode avatar \
  --avatar-config /kaggle/working/dfl-pipeline-stages/private-config.json \
  --gpu-python /tmp/dfl-fullbody-env/bin/python \
  --session-seconds 1800 \
  --token-file /tmp/dfl-avatar-token \
  --evidence-dir /kaggle/working/dfl-avatar-stream-evidence
```

The listener defaults to loopback. HTTPS and the existing validated STUN/TURN configuration are separate operator settings. No tunnel, public endpoint or paid GPU is provisioned by this command. Keep keys out of URLs and logs. The GPU child uses bounded local stdio messages, not another public port. Its upstream initialization logs are redirected away from the binary frame protocol. Startup/render deadlines and strict returned-frame size checks fail closed, with no CPU inference or original-camera passthrough fallback.

The original ten-minute session default remains unchanged. The example explicitly requests thirty minutes and the page shows the countdown. The operator may choose a bounded duration with `--session-seconds`; an HTTPS tunnel's separate shutdown timer is not extended by that option. `--idle-seconds` defaults to thirty seconds.

The server's negotiation timeout now uses `asyncio.wait_for`, which works in the Python 3.10 CPU environment. The first actual test caught the old Python 3.11-only `asyncio.timeout` usage before any successful offer; after correction, actual GPU-backed stop/restart sessions passed. Cancellation drains in-flight rendering before the next session is admitted. GPU frame IDs stay monotonic while each session's RTP/MediaPipe clock resets. A body-tracking failure produces a blank output and explicit tracking status instead of showing the source body or a stale target picture.

`restore_lhmpp_kaggle_runtime.py` also preserves an additional numbered empty-placeholder backup on repeated Kaggle restarts, rather than overwriting an older backup or blocking the same known repair. Unrecognized nonempty files, directories and unrelated symlinks are still rejected without replacement.

## Reproduce transport checks

Actual final T4 results: the two sustained sessions returned 267 nonblank frames in 90.158 seconds and 261 in 90.160 seconds (about 2.96 and 2.89 FPS). Both reported connected/completed ICE and no processing error. Their stale-frame drop counts were 816 and 822; this is expected backpressure at a 12-FPS input rate, not evidence of 12-FPS inference. The GPU child exited with code zero; external nvidia-smi then showed 0 MiB on both T4s. A separate profiled frame recorded 945 CUDA compute events. This establishes a working low-frame-rate loopback avatar transport, not smooth real-time performance.

The opt-in fitted-lip CPU pass completed the existing 180-frame clip with 172 face observations. Its coefficient-vector norm reached the configured cap, and visible mouth excitation remained below the diagnostic threshold. It is not accepted as natural speech lip sync. Tests after the final audit passed 56 Python cases plus the ICE-gathering and OBS-output Node checks. The OBS check validates the DOM contract only, not a Windows device.

`test_lhmpp_stream_roundtrip.py` takes `--config`, `--gpu-python`, `--driver-video`, `--output-dir` and optionally `--hand-diagnostics`. It renders controlled neutral/pucker/funnel/press stills, separately proves CUDA compute, then opens, stops and reopens two authenticated WebRTC sessions. It checks increasing returned timestamps, dimensions and at least 80% nonblank return coverage. Its source is a cycling prerecorded motion stress fixture, not a live laptop webcam.

Use `--minimum-session-seconds 90` for two 90-second loopback sessions. There is one pending frame; stale input is dropped rather than queued. This verifies local transport and lifecycle behavior. It does not verify the user's public network, TURN service, natural motion quality, 30 FPS or indefinite reliability.

## Windows OBS output

1. Connect the preview and click **Open OBS output**. It opens **DeepFaceLive Processed Output**, a same-origin window containing only the returned video.
2. In OBS, add a **Window Capture** source and select that window. Disable capture of the pointer and fit the source to the scene.
3. Click **Start Virtual Camera** in OBS, then select **OBS Virtual Camera** in the receiving application.
4. Keep the preview and output windows open. Stopping/disconnecting clears the separate output. Audio is separate; this video-only preview does not transport microphone audio or promise a synchronized virtual microphone.

Use Window Capture for this window. Opening `/output` as an independent OBS Browser Source does not share the parent browser's MediaStream and therefore cannot display it. The processed-output window is not given the connection key and never selects the local-camera element.

Official references: [OBS Window Capture](https://obsproject.com/kb/window-capture-sources) and [OBS Virtual Camera](https://obsproject.com/kb/virtual-camera-guide). The new output-window DOM contract was tested for cloud-stream selection, same-origin/source validation and clearing on Stop. Actual Windows OBS capture/device behavior is still unverified.

## Hands and stopping rule

`HandObservationGuard` removes global palm rotation before comparing finger observations. It bounds rapid direction changes and sudden segment-length changes; rejection does not poison its last accepted observation. After a long gap it permits reacquisition. It is optional and does not change the existing offline baseline or default hand retargeter.

The saved 180-frame diagnostics produced left 159 accepted / 10 rejected / 11 missing, and right 172 accepted / 1 rejected / 7 missing. These are observation-screening counts, not anatomical ground truth or proof of improved images. Do not present this stage as solved finger reconstruction. Preserve the user's preferred baseline until a visual comparison clearly improves it.

No RunPod service or other paid GPU was started. No source footage, private pose packets, derived model map, token or TURN credential is included in the evidence committed with this stage.
