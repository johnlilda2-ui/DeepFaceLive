# Cloud headless status — 6 November 2026

Active branch: `experiment/cloud-headless-webrtc`.
The existing desktop application and `master` were not modified.

## Verified on Kaggle

Notebook: https://www.kaggle.com/code/johnlilda/deepfacelive/edit

The initial still-image GPU test ran code commit
`1283c45875777c09844915f1fb1c2c23d4219e72`. The worker test ran
`dc2b3ac756e9dab7c14a367f49aafb90b7008ac3`.
These are execution results observed in the notebook, not results inferred from
syntax checks, provider availability, or device selection.

Environment: Python 3.13.15, Tesla T4, driver 580.178.04, ONNX Runtime GPU 1.22.0
(CUDA 12 build), NumPy 2.2.6, ONNX 1.18.0, OpenCV headless 4.11.0.86,
numexpr 2.10.2, h5py 3.12.1. The installed NVIDIA packages included CUDA runtime
12.9.79, cuBLAS 12.9.3.13, and cuDNN 9.28.0.42. The resolved manifest is saved
in `/kaggle/working/dfl-dependencies.txt`.

| Model | Initial smoke test CUDA compute events | Worker CUDA compute events |
| --- | ---: | ---: |
| Face detector | 304 | 1824 |
| Landmarks | 122 | 671 |
| Face swap | 226 | 2260 |
| Identity recognition | 130 | 130 |

Both tests passed with CUDA execution evidence for every model. The initial
swapped face crop was 35,487 bytes. Raw profiles and reports are retained in:

- `/kaggle/working/dfl-gpu-evidence/run-guuqacfo/`
- `/kaggle/working/dfl-worker-results/run-jhen6b0z/`
- `/kaggle/working/dfl-gpu-test.log`
- `/kaggle/working/dfl-worker-test.log`

Kaggle file persistence is enabled so model downloads and run artifacts can
survive the session. The notebook version retains the displayed results.

## Headless worker

`scripts/headless_cloud_worker.py` provides `HeadlessCloudWorker.process_frame()`.
It initializes the models and source embedding once, then processes serial
uint8 BGR frames. Output preserves the frame dimensions, with a feathered ellipse
composite. A detected no-face condition returns an independent unchanged frame;
model/runtime errors propagate. CUDA is required unless `--allow-cpu` is explicit.
No server, browser camera, WebRTC, or paid hosting is started.

The Kaggle integration processed ten repeated 574×574 sample frames, saved full
frames, and passed a real-detector blank-frame test. Warm processing latency:
median 88.88 ms, p95 125.51 ms; first frame 388.72 ms; model/source setup 4.56 s.
These timings exclude disk writes, network, capture, encoding, and OBS. They do
not establish live-video performance or quality under motion/occlusion.

Run the corrected notebook in order; its worker cell is gated by successful
GPU evidence. For the standalone worker in that prepared environment:

```bash
/kaggle/working/dfl-cuda12-env/bin/python scripts/headless_cloud_worker.py \
  --source-image doc/celebs/Rob_Doe/Rob_Doe.png \
  --target-image doc/celebs/Millie_Park/Millie_Park.png \
  --output-dir /kaggle/working/dfl-worker-results --frames 10
```

## Audit and next stage

Six CUDA-evidence/notebook regression tests and five frame/compositing tests
passed locally. The five worker tests also passed on Kaggle. GitHub Actions
passed for worker commit `dc2b3ac756e9dab7c14a367f49aafb90b7008ac3`.

Fixes cover malformed embedded notebook code, CUDA 13 dependency mismatch,
Python 3.13 wheels, Kaggle's unavailable ensurepip bootstrap, missing h5py,
Kaggle diagnostics dependencies, false GPU positives, and no-face/error handling.
Desktop session construction remains unchanged; profiling is scoped to the
standalone test/worker process.

The WebRTC transport stage below is now verified. Next: external Windows
connection, signaling and NAT traversal, then webcam and OBS integration. Use only consent-based inputs. No RunPod or other paid GPU was
started, and paid hosting still requires user approval.


## Verified WebRTC transport — 6 November 2026

`scripts/headless_webrtc_track.py` connects an incoming WebRTC video track to
the existing worker. Inference runs off the event loop. One pending frame is
retained; stale frames are dropped. Source timestamps are preserved, timing
samples are bounded, and shutdown waits for in-flight inference. Processing
failures, including late failures during shutdown, propagate.

The Kaggle test used aiortc 1.15.0 and PyAV 17.0.0. Two actual peer connections
exchanged encoded video over RTP/DTLS, with the T4 worker processing the received
frames and a second WebRTC leg returning them. It did not use a webcam or expose
a public endpoint. The sample image was repeated as a timed video source.

| Check | Observed result |
| --- | --- |
| Peers | Both connected |
| Returned frames | 10, at 574×574 |
| Video packets sent / received | 30 / 30 |
| Incoming frames captured / processed | 37 / 11 at report time |
| Stale frames dropped | 25 |
| Pending frame queue limit | 1 |
| Timestamp order | Strictly increasing |
| Processing p50 / p95 | 89.82 / 138.92 ms |
| Returned vs original pixel MAE | 11.10 |
| Returned vs processed reference pixel MAE | 4.76 |

The decoded return was closer to the processed reference than to the original,
so a connection or unchanged echo alone could not pass. The timing numbers are
worker processing time, not end-to-end webcam/network/OBS latency.

| Model | CUDA compute events in WebRTC run |
| --- | ---: |
| Face detector | 2128 |
| Landmarks | 854 |
| Face swap | 2938 |
| Identity recognition | 130 |

Actual reports and the returned frame are in:

- `/kaggle/working/dfl-webrtc-results/run-xdnrj1td/webrtc-report.json`
- `/kaggle/working/dfl-webrtc-results/run-xdnrj1td/worker-trrjwooa/execution_evidence.json`
- `/kaggle/working/dfl-webrtc-results/run-xdnrj1td/returned_frame.png`
- `/kaggle/working/dfl-webrtc-test.log`

Notebook Step 6 runs this test in the prepared environment. CPU CI runs
`test_webrtc_gpu_roundtrip.py --transport-only`; its report explicitly sets
`gpu_verified` false. The GPU run requires all four execution profiles to show
CUDA compute events. GitHub Actions passed the CPU packet round-trip and all
17 unit/regression tests at commit `0af0498ceecf348963a76f43e81226cc2a94b967`.
Four new transport tests and both environment-repair tests also passed on Kaggle.

Kaggle file persistence restored the original venv's `python` symlink as an empty
file. `prepare_headless_env.py` now uses interpreter copies and repairs only
interpreter aliases in a recognized venv, preserving installed packages.

### Remaining connection stage

The initial loopback test did not prove external connectivity. The subsequent
physical laptop test below now provides external demo evidence and a user
report of webcam face swapping. TURN relay and OBS Virtual Camera remain untested.


## Browser and HTTP signaling stage

The token-protected server, browser camera client, demo mode, Stop/reconnect,
ICE diagnostics, and private TURN configuration are implemented. CPU Chromium
integration and Kaggle GPU HTTP/WebRTC integration passed. Full evidence and the
laptop test procedure are in [windows-cloud-connection.md](windows-cloud-connection.md).

The server remains private by default. The user approved a temporary,
token-protected Cloudflare HTTPS route for the physical laptop test.

## Physical laptop test and preview expiry

The user's screenshots show the fictional demo's original frame and visibly
swapped return, a connected session, 1,018 received frames and 1,025 processed
frames. The selected media path was srflx to srflx, with 258–283 ms reported
network RTT. These counters are not FPS or end-to-end latency measurements.
The user subsequently confirmed that face swapping worked with their webcam;
webcam quality and latency have not been independently measured.

The later screenshot showed HTTP 530 and empty previews. Checking the Kaggle
processes confirmed both the tunnel and GPU server had stopped under the
preview's explicit 20-minute shutdown timer. This is separate from the server's
default 10-minute per-session limit and does not establish a CUDA failure.

The browser now displays the server session countdown, tolerates transient
status/network failures for up to 15 seconds without closing working media,
and explains persistent link failure or expiry. HTTP authentication, missing
session and inference failures still end the session immediately. The temporary
route shutdown timer remains in place; retries cannot renew an expired route.
