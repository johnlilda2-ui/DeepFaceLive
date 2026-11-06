#!/usr/bin/env python3
"""Probe an isolated LHM++ runtime. This is not an avatar inference test.

Run with the experimental Python, --runtime-root pointing to the pinned
upstream checkout. Never install packages in the desktop or face-worker env.
"""

import argparse
import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.runtime_root.resolve(strict=True)
    if not (root / 'core/models/modeling_humana4o_lrm.py').is_file():
        parser.error('--runtime-root must be an LHM++ checkout')
    sys.path.insert(0, str(root))
    report = {
        'scope': 'runtime_only_not_model_inference',
        'python': platform.python_version(),
        'executable': sys.executable,
        'full_body_inference_verified': False,
        'imports': {},
        'gpu_compute_verified': False,
    }
    try:
        report['upstream_commit'] = subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'],
            text=True, timeout=10,
        ).strip()
        import torch
        report.update(torch=torch.__version__, cuda_build=torch.version.cuda)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA initialization failed; CPU fallback is forbidden')
        torch.manual_seed(0)
        x = torch.randn(256, 256, device='cuda')
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]) as prof:
            y = x @ x
            torch.cuda.synchronize()
        kernels = [e.name for e in prof.events()
                   if e.device_type == torch.autograd.DeviceType.CUDA]
        if not kernels or not y.is_cuda or not torch.isfinite(y).all().item():
            raise RuntimeError('No finite CUDA result with executed CUDA events')
        torch.testing.assert_close(y.cpu(), x.cpu() @ x.cpu(), rtol=1e-3, atol=1e-3)
        report.update(gpu_compute_verified=True,
                      gpu=torch.cuda.get_device_name(0), cuda_events=kernels)
    except Exception:
        report['gpu_error'] = traceback.format_exc()
    for name in ('accelerate', 'diffusers', 'transformers', 'xformers.ops',
                 'smplx', 'pointops_cuda', 'spconv.pytorch', 'torch_scatter',
                 'pytorch3d.ops', 'diff_gaussian_rasterization', 'simple_knn._C',
                 'gsplat', 'core.models'):
        try:
            importlib.import_module(name)
            report['imports'][name] = {'ok': True}
        except Exception:
            report['imports'][name] = {'ok': False, 'error': traceback.format_exc()}
    ready = report['gpu_compute_verified'] and all(
        item['ok'] for item in report['imports'].values())
    report['runtime_probe_passed'] = ready
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if ready else 1


if __name__ == '__main__':
    raise SystemExit(main())
