# Full-body avatar feasibility audit

Date: 2026-11-06
Project: johnlilda2-ui/DeepFaceLive
Development branch: experiment/cloud-headless-webrtc
Scope: read-only upstream audit and planning. No avatar packages were installed, model inference was not run, and the existing application/face worker was not changed.

## User requirement

A realistic target's complete face, hair, body and clothing should move with the user's webcam motion, including speech-related facial movement, and return to Windows OBS. A mostly static portrait or face-only swap does not satisfy this requirement. Use fictional examples or consented references.

## Decision

Conditional GO for an isolated, short offline rendering feasibility test of the published LHMPP-700M model on Kaggle T4. NO-GO for promising or integrating live full-body video yet. No paid GPU, card, subscription or hosting was started for this audit.

## Audited upstream and checkpoints

- Code: https://github.com/aigc3d/LHM-plusplus
- Audited code commit: 906b5d9fb967ab42efb92f6fa55bf22cac86b653
- Published model: https://huggingface.co/3DAIGC/LHMPP-700M
- Model revision: fc9f73664b9bfcc457e96210ddc06fe7caf0d559
- Public metadata returned HTTP 200, gated=false; files include config.json and model.safetensors.
- Retrieved config.json. An actual byte-range request for model.safetensors returned HTTP 206, bytes 0-7/4527981124; safetensors header length 163696. This verifies file access, not full-file integrity or inference.
- Public prior-assets metadata: https://huggingface.co/3DAIGC/LHMPP-Prior
- Prior revision: b683c8f68bede4f318b0bb539730b8e6711d30a0; metadata HTTP 200, gated=false. Prior weights were not downloaded or executed.
- Anonymous Hugging Face metadata requests for LHMPP-700M-PixelShuffle, LHMPP-700M-SMPLX-FREE and LHMPPS-700M returned HTTP 401. This does not distinguish unpublished/private repositories from access restrictions. Upstream also explicitly marks PixelShuffle hub weights as pending. ModelScope access was not verified; do not assume those variants are obtainable.
- The published LHMPP-700M is not in upstream GS_RENDER_SUPPORTED_MODEL_NAMES. Do not promise the newer plain-Gaussian export/rendering path for this checkpoint.

## Software and T4 compatibility

Upstream INSTALL.md specifies Linux, Python 3.10, PyTorch 2.3.0 and CUDA 12.1. requirements.txt includes older compiled dependencies (pointops, spconv, torch_scatter, PyTorch3D, Gaussian rasterizers, gsplat, xformers). Kaggle's current Python 3.13 host cannot simply use these pinned Python 3.10 wheels. A separate Python 3.10 environment is needed; do not replace the working face-swap CUDA environment or Kaggle host packages.

The retrieved checkpoint config uses DINOv2, not Sapiens, for its image encoder. Optional Sapiens code uses BF16 but is not selected by this checkpoint; do not describe it as a confirmed blocker for LHMPP-700M. The point-transformer config enables FlashAttention, but core/models/encoders/sonata/model.py contains a non-FlashAttention fallback when the package is absent. T4 support, peak memory and numerical behavior of that fallback remain untested. The upstream 8 GB memory and sub-second reconstruction claims are not measurements of our T4, live motion tracking, speech animation or complete streaming latency.

## Motion and face tracking gaps

The app renders prepared motion sequences. A reference-image pose estimator exists in engine/pose_estimation/pose_estimator.py, but this is not an end-to-end live webcam streaming integration. The offline helper predict_motion_seqs_from_images invokes thirdparty/multi-hmr; .gitmodules points that submodule to an internal upstream GitLab host. Do not attempt private access or assume recursive cloning produces a runnable tracker. The supplied public pose-estimation code needs separate evaluation as a possible alternative.

SMPL-X/FLAME motion parameters include body, hands, jaw and expression. Their presence in the renderer is not proof of live lip tracking, accurate identity, temporal stability or correct synchronization. A compatible live tracker and expression retargeting path remain separate tasks.

## Licensing

Upstream source code LICENSE is Apache-2.0. LICENSE_WEIGHT is Creative Commons Attribution-NonCommercial 4.0; do not call the entire pipeline unrestricted for commercial use. LICENSE_NVIDIA contains a research/evaluation noncommercial use limitation for covered components. SMPL-X, FLAME, prior assets and other dependencies require their own provenance/license checks. Personal noncommercial evaluation is the proposed scope, not a cleared commercial deployment.

Official references:
- https://github.com/aigc3d/LHM-plusplus/blob/main/LICENSE_WEIGHT
- https://github.com/aigc3d/LHM-plusplus/blob/main/LICENSE_NVIDIA
- https://github.com/aigc3d/LHM-plusplus/blob/main/INSTALL.md
- https://github.com/aigc3d/LHM-plusplus/blob/main/core/utils/model_card.py

## Next bounded experiment

1. Check storage, available CUDA compiler and feasibility of an isolated Python 3.10 environment on Kaggle. Review only the selected checkpoint's required assets rather than running download_all.py.
2. Pin code, checkpoint revisions and dependencies. Review downloaded custom-extension sources before building. Record full-file checksums if weights are downloaded.
3. Use a fictional/consented full-body reference and a supplied prerecorded pose sequence. First render a single frame, then a roughly 3-second low-resolution sequence, with explicit timeout and memory limits. Failures are results; do not silently fall back to CPU or claim GPU success from provider availability.
4. Record actual GPU execution, peak memory, setup time, reconstruction time, rendering FPS and a visibly reviewed output. Judge target likeness, clothing, body proportions, hands and temporal artifacts.
5. Proceed to a live tracker only if rendering quality and speed justify it. Benchmark tracking, expression retargeting and transport together before integrating with the existing WebRTC/OBS path.

No full-body live capability, T4 FPS, exact resemblance or speech synchronization has been verified.
