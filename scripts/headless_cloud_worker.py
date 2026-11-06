#!/usr/bin/env python3
"""Separate, serial headless frame worker for consent-based face swapping.

Loads models and the source embedding once. This stage has no network listener,
WebRTC, camera, GUI, or hosting API. The CLI benchmarks repeated still frames;
process_frame() is the transport-independent entry point for later video work.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import headless_inference_smoke_test as smoke


def validate_frame(frame):
    import numpy as np
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or
            frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 32):
        raise ValueError('Expected a uint8 BGR frame, HxWx3, at least 32x32.')


def composite_face(frame, swapped, transform):
    """Feathered ellipse baseline, using premultiplied alpha to avoid dark edges."""
    import cv2
    import numpy as np
    height, width = frame.shape[:2]
    size = swapped.shape[0]
    alpha = np.zeros((size, size), dtype=np.float32)
    cv2.ellipse(alpha, (size // 2, size // 2), (int(size * .38), int(size * .43)),
                0, 0, 360, 1.0, -1)
    alpha = cv2.GaussianBlur(alpha, (21, 21), 5)
    matrix = transform.invert().to_exact_mat(size, size, width, height)
    warped_alpha = cv2.warpAffine(alpha, matrix, (width, height), flags=cv2.INTER_LINEAR)
    premultiplied = swapped.astype(np.float32) * alpha[..., None]
    warped_face = cv2.warpAffine(premultiplied, matrix, (width, height), flags=cv2.INTER_LINEAR)
    output = frame.astype(np.float32) * (1.0 - warped_alpha[..., None]) + warped_face
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


class HeadlessCloudWorker:
    """One source identity, one in-flight frame; create one instance per worker."""
    def __init__(self, source_image, evidence_dir, require_cuda=True, threshold=.5):
        import cv2
        self.threshold = threshold
        self.require_cuda = require_cuda
        self.frame_count = 0
        self.closed = False
        evidence_dir = Path(evidence_dir).resolve()
        evidence_dir.mkdir(parents=True, exist_ok=True)
        self.profile_dir = Path(tempfile.mkdtemp(prefix='worker-', dir=evidence_dir))
        self.device = smoke.choose_device(require_cuda)
        smoke.validate_and_assemble_models(smoke.REPO_ROOT)
        with smoke.profiled_sessions(self.profile_dir, require_cuda):
            self.models = smoke.load_models(self.device, require_cuda)
        self.detector, self.marker, self.swap_model = self.models
        source = smoke.load_image(str(source_image), 'source', cv2)
        aligned = smoke.detect_and_align_face(source, 'source', self.detector, self.marker,
                                             1.55, 112, self.threshold)
        self.face_vector = smoke.get_valid_face_vector(self.swap_model, aligned)

    def process_frame(self, frame):
        """Return a full-size BGR frame and status; no-face frames pass through."""
        import numpy as np
        if self.closed:
            raise RuntimeError('Worker is closed.')
        validate_frame(frame)
        self.frame_count += 1
        try:
            aligned, transform = smoke.detect_and_align_face(
                frame, 'target', self.detector, self.marker, 2.2, 128,
                self.threshold, return_transform=True)
        except smoke.NoFaceDetected:
            return frame.copy(), 'no_face'
        swapped = self.swap_model.generate(aligned, self.face_vector)
        if (swapped is None or swapped.shape != aligned.shape or
                swapped.dtype != np.uint8 or not np.isfinite(swapped).all()):
            raise RuntimeError('Swap model returned an invalid frame.')
        return composite_face(frame, swapped, transform), 'swapped'

    def close(self):
        """Finalize model profiles once, including proof of actual CUDA execution."""
        if self.closed:
            return
        self.closed = True
        smoke.save_execution_evidence(self.models, self.profile_dir, self.require_cuda)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-image', required=True)
    parser.add_argument('--target-image', required=True)
    parser.add_argument('--output-dir', default='artifacts/headless-worker')
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--allow-cpu', action='store_true', help='Explicit diagnostic CPU mode.')
    args = parser.parse_args()
    if not 1 <= args.frames <= 10000:
        parser.error('--frames must be between 1 and 10000')
    import cv2
    import numpy as np
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Unique run outputs prevent old successful files from being mistaken for a new run.
    run_dir = Path(tempfile.mkdtemp(prefix='run-', dir=output_dir))
    started = time.perf_counter()
    worker = HeadlessCloudWorker(args.source_image, run_dir, require_cuda=not args.allow_cpu)
    setup_ms = (time.perf_counter() - started) * 1000
    target = smoke.load_image(args.target_image, 'target', cv2)
    latencies = []
    for index in range(args.frames):
        started = time.perf_counter()
        processed, status = worker.process_frame(target)
        elapsed = (time.perf_counter() - started) * 1000
        if status != 'swapped':
            raise RuntimeError('Benchmark target must contain a detectable face.')
        if processed.shape != target.shape or processed.dtype != target.dtype:
            raise RuntimeError('Worker changed the full-frame shape or dtype.')
        difference = float(np.abs(processed.astype(np.int16) - target.astype(np.int16)).mean())
        if difference == 0:
            raise RuntimeError('Processed frame is unchanged.')
        path = run_dir / f'frame-{index:05d}.png'
        if not cv2.imwrite(str(path), processed):
            raise RuntimeError(f'Failed to write {path}')
        latencies.append(elapsed)
        print(f'FRAME {index}: {status}, {elapsed:.2f} ms, mean pixel change {difference:.3f}', flush=True)
    # Exercise the normal no-face path with a real detector; inference failures propagate.
    blank = np.zeros_like(target)
    passthrough, status = worker.process_frame(blank)
    if status != 'no_face' or not np.array_equal(passthrough, blank):
        raise RuntimeError('No-face frame did not pass through unchanged.')
    worker.close()
    warm = latencies[1:] or latencies
    report = {'status': 'passed', 'benchmark': 'repeated still frame, not live video',
              'setup_ms': setup_ms, 'frames': args.frames, 'latency_ms': latencies,
              'warm_p50_ms': float(np.percentile(warm, 50)),
              'warm_p95_ms': float(np.percentile(warm, 95)),
              'no_face_passthrough': True, 'output_shape': list(target.shape),
              'execution_evidence': str(worker.profile_dir / 'execution_evidence.json')}
    report_path = run_dir / 'worker-report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    print(f'HEADLESS WORKER TEST PASSED: {report_path}', flush=True)


if __name__ == '__main__':
    main()
