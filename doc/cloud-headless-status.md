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

Next: a separate WebRTC transport stage, followed by local webcam and OBS
integration. Use only consent-based inputs. No RunPod or other paid GPU was
started, and paid hosting still requires user approval.
