#!/usr/bin/env python3
"""Offline LHM++ frame/short-clip probe; requires prepared, verified assets.

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
    parser.add_argument('--t4-memory-efficient-attention', action='store_true')
    parser.add_argument('--reuse-render-allocator', action='store_true')
    parser.add_argument('--frames', type=int, default=1)
    parser.add_argument('--motion-stride', type=int, default=5)
    parser.add_argument('--fps', type=int, default=6)
    parser.add_argument('--render-height', type=int, default=420)
    parser.add_argument('--framing-box', type=float, nargs=4, metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'))
    parser.add_argument('--benchmark-repeats', type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.frames <= 18 or args.motion_stride < 1 or not 1 <= args.fps <= 30:
        parser.error('Require 1..18 frames, positive motion stride, and 1..30 fps')
    if args.render_height < 14 or args.render_height > 840 or args.render_height % 14:
        parser.error('--render-height must be a multiple of 14 between 14 and 840')
    if not 0 <= args.benchmark_repeats <= 5:
        parser.error('--benchmark-repeats must be between 0 and 5')
    root = args.runtime_root.resolve(strict=True)
    model_dir = args.model_dir.resolve(strict=True)
    reference = args.reference.resolve(strict=True)
    pose_json = args.pose_json.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'offline_neural_render_probe', 'passed': False,
              'live_verified': False, 'reference': reference.name,
              'pose_json': str(pose_json), 'stage': 'preflight',
              'requested_frames': args.frames, 'output_fps': args.fps,
              'motion_stride': args.motion_stride}
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
        if not all(p.is_cuda for p in model.parameters()):
            raise RuntimeError('All model parameters must reside on CUDA')
        if args.t4_memory_efficient_attention:
            from lhmpp_t4_attention import enable_t4_attention
            report['attention_adapter'] = enable_t4_attention(model)
        if args.reuse_render_allocator:
            from lhmpp_render_allocator import enable_render_allocator_reuse
            report['allocator_adapter'] = enable_render_allocator_reuse(model, torch)
        helper_path = root / 'scripts/inference/to_gs_ply.py'
        spec = importlib.util.spec_from_file_location('lhmpp_pose_helpers', helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        available = sorted(pose_json.parent.glob('*.json'))
        first = available.index(pose_json)
        selected = available[first::args.motion_stride][:args.frames]
        if len(selected) != args.frames:
            raise RuntimeError('Not enough prepared motion frames')
        sequences = [helper._build_motion_seq_from_pose_json(str(path), {}) for path in selected]
        motion = sequences[0]
        if any(seq['ori_size'] != motion['ori_size'] for seq in sequences):
            raise RuntimeError('Changing motion image size is unsupported')
        for key in ('render_c2ws', 'render_intrs', 'render_bg_colors'):
            motion[key] = torch.cat([seq[key] for seq in sequences], dim=1)
        motion['smplx_params'] = {
            key: value if key == 'betas' else torch.cat(
                [seq['smplx_params'][key] for seq in sequences], dim=1)
            for key, value in sequences[0]['smplx_params'].items()}
        motion['offset_list'] = [seq['offset_list'][0] for seq in sequences]
        report['motion_frame_files'] = [path.name for path in selected]
        old_h, old_w = motion['ori_size']
        from lhmpp_camera_framing import framing_transform, transform_intrinsics
        framing = framing_transform(old_w, old_h, args.render_height, args.framing_box)
        report['camera_framing'] = framing
        width, height = framing['width'], framing['height']
        params = motion['smplx_params']
        intrinsics = motion['render_intrs']
        torch.testing.assert_close(intrinsics[..., 0, 0], params['focal'][..., 0])
        torch.testing.assert_close(intrinsics[..., 1, 1], params['focal'][..., 1])
        torch.testing.assert_close(intrinsics[..., :2, 2], params['princpt'])
        if torch.count_nonzero(intrinsics[..., 0, 1]) or torch.count_nonzero(intrinsics[..., 1, 0]):
            raise RuntimeError('Skewed motion cameras are unsupported')
        focal, principal = transform_intrinsics(
            [params['focal'][..., 0], params['focal'][..., 1]],
            [params['princpt'][..., 0], params['princpt'][..., 1]], framing)
        params['focal'] = torch.stack(focal, dim=-1)
        params['princpt'] = torch.stack(principal, dim=-1)
        intrinsics[..., 0, 0], intrinsics[..., 1, 1] = focal
        intrinsics[..., 0, 2], intrinsics[..., 1, 2] = principal
        params['img_size_wh'][...] = torch.tensor([width, height])
        params['betas'] = torch.from_numpy(betas).float().reshape(1, -1)
        motion['ori_size'] = (height, width)
        # The upstream no-mask branch infers dimensions from 2*principal_point,
        # which is invalid for a calibrated off-centre crop. Its canvas/mask
        # branch accepts explicit dimensions; values are unused in that path.
        # These full-canvas arrays are size carriers, not segmentation results.
        motion['masks'] = [np.full((height, width), 255, dtype=np.uint8)
                           for _ in selected]
        motion['offset_list'] = [[1.0, 1.0, 0, 0] for _ in selected]
        report['explicit_render_canvas'] = [height, width]
        tensor = torch.from_numpy(np.asarray(source).copy()).permute(2, 0, 1).float()[None] / 255
        report['stage'] = 'neural_inference'
        print('STAGE neural_inference', flush=True)
        timings = {'reconstruction': [], 'render': []}
        reconstruction_cache = {}
        original_reconstruct = model.infer_single_view
        def reconstruct(*fn_args, **fn_kwargs):
            value = original_reconstruct(*fn_args, **fn_kwargs)
            reconstruction_cache['value'] = value
            return value
        def timed(label, fn):
            def call(*fn_args, **fn_kwargs):
                torch.cuda.synchronize()
                begin = time.monotonic()
                value = fn(*fn_args, **fn_kwargs)
                torch.cuda.synchronize()
                timings[label].append(time.monotonic() - begin)
                return value
            return call
        model.infer_single_view = timed('reconstruction', reconstruct)
        model.animation_infer = timed('render', model.animation_infer)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.inference_mode(), torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
        ]) as prof:
            frames = helper._ai.inference_results(
                model, tensor, params, motion, video_size=args.frames,
                batch_size=1, device='cuda', infer_output_renderer='neural')
            torch.cuda.synchronize()
        report['inference_seconds'] = time.monotonic() - start
        kernels = [e.name for e in prof.events()
                   if e.device_type == torch.autograd.DeviceType.CUDA
                   and not e.name.startswith(('Memcpy', 'Memset'))]
        prof.export_chrome_trace(str(output / 'cuda-trace.json'))
        render_seconds = sum(timings['render'])
        report.update(cuda_kernel_count=len(kernels), cuda_kernel_examples=kernels[:20],
                      reconstruction_seconds=sum(timings['reconstruction']),
                      render_seconds=render_seconds,
                      render_fps_excluding_reconstruction=args.frames / render_seconds,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if (not kernels or frames.shape != (args.frames, height, width, 3)
                or not np.isfinite(frames).all()):
            raise RuntimeError('Missing CUDA events or invalid frame output')
        if args.benchmark_repeats:
            # Reuse the identical reference reconstruction only within this
            # fixed-reference, fixed-pose benchmark. Never cache across uploads.
            model.infer_single_view = lambda *a, **kw: reconstruction_cache['value']
            def render_cached():
                return helper._ai.inference_results(
                    model, tensor, params, motion, video_size=args.frames,
                    batch_size=1, device='cuda', infer_output_renderer='neural')
            with torch.inference_mode():
                render_cached()  # one explicit warmup, excluded from measurements
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                samples = []
                for repeat in range(args.benchmark_repeats):
                    timings['render'].clear()
                    torch.cuda.synchronize()
                    begin = time.monotonic()
                    repeated = render_cached()
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - begin
                    error = float(np.abs(repeated.astype(float) - frames.astype(float)).mean())
                    if not np.isfinite(repeated).all() or error > 0.5:
                        raise RuntimeError('Cached rendering changed output unexpectedly')
                    samples.append({'sequence_seconds': elapsed,
                                    'fps_including_cpu_and_readback': args.frames / elapsed,
                                    'render_call_seconds': sum(timings['render']),
                                    'mean_abs_difference_from_verified_run': error})
            report['unprofiled_cached_benchmark'] = {
                'warmup_sequences': 1, 'repeats': samples,
                'median_fps_including_cpu_and_readback': float(np.median(
                    [sample['fps_including_cpu_and_readback'] for sample in samples])),
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'reconstruction_excluded': True, 'video_encoding_excluded': True,
                'live_tracking_and_network_excluded': True}
        np.save(output / 'frames.npy', frames, allow_pickle=False)
        frame = frames[0]
        if frame.std() < 1:
            raise RuntimeError('Frame is effectively blank; visual verification required')
        Image.fromarray(frame).save(output / 'frame.png')
        if args.frames > 1:
            import imageio.v3 as iio
            iio.imwrite(output / 'motion.mp4', frames, fps=args.fps, codec='libx264',
                        pixelformat='yuv420p', macro_block_size=2)
            change = np.abs(frames[-1].astype(float) - frames[0].astype(float)).mean()
            report['first_last_frame_mean_abs_difference'] = float(change)
            if change < 0.1:
                raise RuntimeError('Motion output is effectively static')
        report.update(passed=True, stage='rendered_requires_visual_review',
                      frame_shape=list(frame.shape), frame_std=float(frame.std()))
    except Exception:
        report['error'] = traceback.format_exc()
        if 'torch' in locals() and torch.cuda.is_initialized():
            report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

