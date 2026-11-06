#!/usr/bin/env python3
"""Offline LHM++ neural-render probe; requires prepared, verified model assets.

Use fictional or consented full-body references. This does not implement live
tracking, streaming, lip sync, or guarantee target likeness.
"""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

UPSTREAM_COMMIT = '906b5d9fb967ab42efb92f6fa55bf22cac86b653'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime-root', 'model-dir', 'reference', 'pose-json', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    root = args.runtime_root.resolve(strict=True)
    model_dir = args.model_dir.resolve(strict=True)
    reference = args.reference.resolve(strict=True)
    pose_json = args.pose_json.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'offline_single_frame_neural_render', 'passed': False,
              'live_verified': False, 'reference': reference.name,
              'pose_json': str(pose_json), 'stage': 'preflight'}
    os.chdir(root)
    sys.path.insert(0, str(root))
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                           timeout=10).strip()
        if revision != UPSTREAM_COMMIT:
            raise RuntimeError('Unexpected upstream code revision: ' + revision)
        report['upstream_commit'] = revision
        import numpy as np
        import torch
        from accelerate import Accelerator
        from PIL import Image, ImageOps
        from safetensors.torch import load_file
        torch._dynamo.config.disable = True
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; CPU fallback is forbidden')
        Accelerator()
        report.update(gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
        source = ImageOps.pad(Image.open(reference).convert('RGB'), (504, 840),
                              color='white', method=Image.Resampling.LANCZOS)
        source.save(output / 'reference.png')
        report['stage'] = 'reference_shape'
        print('STAGE reference_shape', flush=True)
        from engine.pose_estimation.pose_estimator import PoseEstimator
        estimator = PoseEstimator('./pretrained_models/human_model_files', device='cuda')
        with torch.inference_mode():
            shape = estimator(np.asarray(source))
        if not shape.is_full_body:
            raise RuntimeError('Reference shape estimation rejected image: ' + str(shape))
        betas = np.asarray(shape.beta).copy()
        report['reference_betas'] = betas.tolist()
        del estimator, shape
        gc.collect()
        torch.cuda.empty_cache()
        report['stage'] = 'strict_model_load'
        print('STAGE strict_model_load', flush=True)
        from core.models.modeling_humana4o_lrm import ModelHumanA4OLRM
        config = json.loads((model_dir / 'config.json').read_text())
        # The published inference checkpoint omits the training identity-loss net.
        # Audited upstream only references it from train()/the training runner.
        config['use_face_id'] = False
        model = ModelHumanA4OLRM(**config)
        weights = load_file(str(model_dir / 'model.safetensors'), device='cpu')
        # This exact decoder inherits unused transformer blocks, but its overridden
        # predict_rgbs() uses patchify -> chunk -> dense_predictor exclusively.
        # Remove only this dead inference branch; strict loading still covers all
        # active parameters. Reject a different decoder or checkpoint layout.
        from core.models.transformer_block.dpt_decoder import PatchDPT4DecoderOnly
        if type(model.neural_renderer) is not PatchDPT4DecoderOnly:
            raise RuntimeError('The audited inference-only adapter requires PatchDPT4DecoderOnly')
        prefix = 'neural_renderer.transformer_block.'
        if any(key.startswith(prefix) for key in weights):
            raise RuntimeError('Checkpoint contains transformer weights; do not prune them')
        model.neural_renderer.transformer_block = torch.nn.ModuleList()
        report['omitted_training_only_modules'] = ['id_face_net', prefix.rstrip('.')]
        model.load_state_dict(weights, strict=True)
        del weights
        gc.collect()
        report['strict_weights_loaded'] = True
        # Upstream train() overrides nn.Module.train without returning self, so
        # eval() also returns None. Do not chain it with device placement.
        model.eval()
        model.to('cuda')
        helper_path = root / 'scripts/inference/to_gs_ply.py'
        spec = importlib.util.spec_from_file_location('lhmpp_pose_helpers', helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        motion = helper._build_motion_seq_from_pose_json(str(pose_json), {})
        old_h, old_w = motion['ori_size']
        height = 420
        width = max(14, round(height * old_w / old_h / 14) * 14)
        sx, sy = width / old_w, height / old_h
        motion['render_intrs'][..., 0, :] *= sx
        motion['render_intrs'][..., 1, :] *= sy
        params = motion['smplx_params']
        params['focal'] *= torch.tensor([sx, sy])
        params['princpt'] *= torch.tensor([sx, sy])
        params['img_size_wh'][...] = torch.tensor([width, height])
        params['betas'] = torch.from_numpy(betas).float().reshape(1, -1)
        motion['ori_size'] = (height, width)
        tensor = torch.from_numpy(np.asarray(source).copy()).permute(2, 0, 1).float()[None] / 255
        report['stage'] = 'neural_inference'
        print('STAGE neural_inference', flush=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.inference_mode(), torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
        ]) as prof:
            frames = helper._ai.inference_results(
                model, tensor, params, motion, video_size=1,
                batch_size=1, device='cuda', infer_output_renderer='neural')
            torch.cuda.synchronize()
        report['inference_seconds'] = time.monotonic() - start
        kernels = [e.name for e in prof.events()
                   if e.device_type == torch.autograd.DeviceType.CUDA]
        report.update(cuda_event_count=len(kernels), cuda_event_examples=kernels[:20],
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if not kernels or frames.shape[0] != 1 or not np.isfinite(frames).all():
            raise RuntimeError('Missing CUDA events or invalid frame output')
        frame = frames[0]
        if frame.std() < 1:
            raise RuntimeError('Frame is effectively blank; visual verification required')
        Image.fromarray(frame).save(output / 'frame.png')
        report.update(passed=True, stage='rendered_requires_visual_review',
                      frame_shape=list(frame.shape), frame_std=float(frame.std()))
    except Exception:
        report['error'] = traceback.format_exc()
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
