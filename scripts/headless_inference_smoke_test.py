#!/usr/bin/env python3
"""Headless DeepFaceLive inference smoke test.

Run from the repository root:
  python scripts/headless_inference_smoke_test.py --source-image source.jpg --target-image target.jpg

The output is a swapped, aligned face crop (not a full-frame composite or live stream).
No GUI, camera, web server, or paid cloud GPU is started by this script.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class SmokeTestFailure(RuntimeError):
    """Raised after a smoke-test stage prints its failure details."""


def run_stage(label: str, fn: Callable[[], Any]) -> Any:
    print(f"\n[RUN ] {label}", flush=True)
    started = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print(
            f"[FAIL] {label} ({elapsed:.2f}s): {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        raise SmokeTestFailure(label) from exc
    print(f"[PASS] {label} ({time.perf_counter() - started:.2f}s)", flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check DeepFaceLive's headless ONNX face-swap inference path."
    )
    parser.add_argument("--source-image", required=True, help="Image containing the identity to use.")
    parser.add_argument("--target-image", required=True, help="Image containing the face to swap.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output swapped-face crop (default: artifacts/headless-smoke-test/swapped_face.png).",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Face detection confidence threshold.")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail rather than use CPU if CUDAExecutionProvider or a CUDA device is unavailable.",
    )
    return parser.parse_args()


def choose_device(require_cuda: bool):
    import onnxruntime as ort
    from xlib.onnxruntime import get_available_devices_info, get_cpu_device_info

    providers = ort.get_available_providers()
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"Available execution providers: {', '.join(providers)}")

    try:
        devices = get_available_devices_info()
    except Exception:
        if require_cuda:
            raise
        print("[WARN] Project GPU discovery failed; attempting CPU fallback.")
        traceback.print_exc()
        devices = [get_cpu_device_info()]

    cuda_device = next(
        (device for device in devices if device.get_execution_provider() == "CUDAExecutionProvider"),
        None,
    )
    cuda_provider_present = "CUDAExecutionProvider" in providers

    if cuda_provider_present and cuda_device is not None:
        print(f"Selected device: {cuda_device}")
        return cuda_device

    reason = (
        "CUDAExecutionProvider is not installed in this ONNX Runtime build."
        if not cuda_provider_present
        else "ONNX Runtime lists CUDA, but the project did not discover a usable CUDA device."
    )
    if require_cuda:
        raise RuntimeError(reason)
    print(f"[WARN] {reason} Using CPU for this smoke test.")
    cpu_device = next(
        (device for device in devices if device.get_execution_provider() == "CPUExecutionProvider"),
        None,
    )
    if cpu_device is None:
        cpu_device = get_cpu_device_info()
    print(f"Selected device: {cpu_device}")
    return cpu_device


def validate_and_assemble_models(root: Path) -> dict[str, Path]:
    from xlib.file import SplittedFile

    swap_dir = root / "modelhub" / "onnx" / "InsightFaceSwap"
    model_specs = [
        ("inswapper_128.onnx", swap_dir),
        ("w600k_r50.onnx", swap_dir),
    ]
    result: dict[str, Path] = {}

    for filename, directory in model_specs:
        model_path = directory / filename
        part_paths = []
        for candidate in directory.glob(filename + ".part*"):
            match = re.fullmatch(re.escape(filename) + r"\.part(\d+)", candidate.name)
            if match:
                part_paths.append((int(match.group(1)), candidate))
        part_paths.sort(key=lambda item: item[0])

        if part_paths:
            indexes = [index for index, _ in part_paths]
            expected_indexes = list(range(len(part_paths)))
            if indexes != expected_indexes:
                raise RuntimeError(
                    f"Split model parts for {model_path} are not continuous: "
                    f"found {indexes}, expected {expected_indexes}."
                )
            expected_size = sum(path.stat().st_size for _, path in part_paths)
            if model_path.exists() and model_path.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Existing assembled model has the wrong size: {model_path} is "
                    f"{model_path.stat().st_size} bytes, but its parts total "
                    f"{expected_size} bytes. Move or remove the stale assembled file, "
                    "then rerun this test to reassemble it."
                )
            if not model_path.exists():
                print(f"Assembling {model_path.name} from {len(part_paths)} parts...")
                SplittedFile.merge(model_path, delete_parts=False)
            if not model_path.exists() or model_path.stat().st_size != expected_size:
                actual = model_path.stat().st_size if model_path.exists() else "missing"
                raise RuntimeError(
                    f"Model assembly validation failed for {model_path}: "
                    f"expected {expected_size} bytes, got {actual}."
                )
        elif not model_path.is_file() or model_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Missing model {model_path} and no continuous .part0, .part1, ... files were found."
            )

        if model_path.stat().st_size == 0:
            raise RuntimeError(f"Model file is empty: {model_path}")
        result[filename] = model_path
        print(f"Validated {model_path.relative_to(root)} ({model_path.stat().st_size:,} bytes)")

    for relative_path in (
        "modelhub/onnx/YoloV5Face/YoloV5Face.onnx",
        "modelhub/onnx/InsightFace2d106/InsightFace2D106.onnx",
    ):
        path = root / relative_path
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required detector/landmark model is missing or empty: {path}")
        print(f"Validated {relative_path} ({path.stat().st_size:,} bytes)")
    return result


def load_image(path_text: str, role: str, cv2):
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{role} image does not exist: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode the {role} image: {path}")
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] < 32 or image.shape[1] < 32:
        raise ValueError(f"Invalid {role} image shape {image.shape} for {path}")
    print(f"{role.title()} image: {path} ({image.shape[1]}x{image.shape[0]})")
    return image


def detect_and_align_face(image, role: str, detector, marker, coverage: float, output_size: int, threshold: float):
    import numpy as np
    from xlib.face import ELandmarks2D, FLandmarks2D, FRect

    height, width = image.shape[:2]
    detections = detector.extract(image, threshold=threshold)[0]
    if len(detections) == 0:
        raise RuntimeError(
            f"No face detected in the {role} image. Use a clear, front-facing image with one visible face."
        )

    rectangles = [
        FRect.from_ltrb((float(left) / width, float(top) / height,
                         float(right) / width, float(bottom) / height))
        for left, top, right, bottom in detections
    ]
    face_rect = FRect.sort_by_area_size(rectangles)[0]
    marker_crop, crop_matrix = face_rect.cut(image, 1.6, 192)
    landmarks_batch = marker.extract(marker_crop)
    if landmarks_batch is None or len(landmarks_batch) == 0:
        raise RuntimeError(f"Landmark model returned no landmarks for the {role} face.")

    landmarks = np.asarray(landmarks_batch[0])[..., 0:2] / np.asarray((192.0, 192.0), dtype=np.float32)
    if landmarks.shape != (106, 2) or not np.isfinite(landmarks).all():
        raise RuntimeError(
            f"Unexpected landmark output for {role}: shape={landmarks.shape}, "
            f"finite={np.isfinite(landmarks).all()}."
        )
    global_landmarks = FLandmarks2D.create(ELandmarks2D.L106, landmarks).transform(
        crop_matrix, invert=True
    )
    aligned_face, _ = global_landmarks.cut(
        image, coverage=coverage, output_size=output_size, y_offset=-0.15
    )
    if aligned_face is None or aligned_face.size == 0:
        raise RuntimeError(f"Alignment returned an empty face crop for the {role} image.")
    return aligned_face


def main() -> int:
    args = parse_args()
    root = REPO_ROOT
    if not (root / "modelhub" / "onnx").is_dir():
        print(f"[FAIL] Repository root is not valid: {root}", file=sys.stderr)
        return 2

    try:
        device = run_stage("Check ONNX Runtime providers and select device", lambda: choose_device(args.require_cuda))
        run_stage("Assemble and validate required model files", lambda: validate_and_assemble_models(root))

        images = run_stage(
            "Load source and target images",
            lambda: load_both_images(args),
        )
        source_image, target_image, cv2 = images

        models = run_stage("Load face detector, landmark model, and swap model", lambda: load_models(device))
        detector, marker, swap_model = models

        aligned = run_stage(
            "Detect faces and align source identity and target face",
            lambda: (
                detect_and_align_face(source_image, "source", detector, marker, 1.55, 112, args.threshold),
                detect_and_align_face(target_image, "target", detector, marker, 2.2, 128, args.threshold),
            ),
        )
        source_aligned, target_aligned = aligned

        face_vector = run_stage(
            "Run source identity embedding inference",
            lambda: get_valid_face_vector(swap_model, source_aligned),
        )
        output_image = run_stage(
            "Run face-swap inference and save result",
            lambda: generate_and_save(swap_model, target_aligned, face_vector, args.output, root, cv2),
        )
        print("\nSMOKE TEST PASSED")
        print(f"Selected ONNX device: {device}")
        print(f"Saved swapped aligned face crop: {output_image}")
        print("Note: this verifies still-image inference only; it does not test full-frame compositing or live video latency.")
        return 0
    except SmokeTestFailure:
        print("\nSMOKE TEST FAILED. The failing stage and Python traceback above identify the cause.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[FAIL] Smoke test setup: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


def load_both_images(args: argparse.Namespace):
    import cv2

    source = load_image(args.source_image, "source", cv2)
    target = load_image(args.target_image, "target", cv2)
    return source, target, cv2


def load_models(device):
    from modelhub.onnx import InsightFace2D106, InsightFaceSwap, YoloV5Face

    detector = YoloV5Face(device)
    print("Loaded YOLOv5 face detector.")
    marker = InsightFace2D106(device)
    print("Loaded 106-point landmark model.")
    swap_model = InsightFaceSwap(device)
    print("Loaded InsightFaceSwap and identity-recognition models.")
    return detector, marker, swap_model


def get_valid_face_vector(swap_model, source_aligned):
    import numpy as np

    vector = swap_model.get_face_vector(source_aligned)
    if vector is None or vector.size == 0 or not np.isfinite(vector).all():
        raise RuntimeError("Source identity model returned an empty or non-finite face vector.")
    print(f"Identity vector shape: {vector.shape}")
    return vector


def generate_and_save(swap_model, target_aligned, face_vector, output_arg, root: Path, cv2) -> Path:
    import numpy as np

    swapped = swap_model.generate(target_aligned, face_vector)
    if swapped is None or swapped.size == 0:
        raise RuntimeError("Swap model returned an empty image.")
    if swapped.shape != target_aligned.shape:
        raise RuntimeError(
            f"Unexpected output shape {swapped.shape}; expected {target_aligned.shape}."
        )
    if not np.isfinite(swapped).all():
        raise RuntimeError("Swap model returned NaN or infinite pixel values.")

    output_path = (
        Path(output_arg).expanduser().resolve()
        if output_arg
        else root / "artifacts" / "headless-smoke-test" / "swapped_face.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), swapped):
        raise RuntimeError(f"OpenCV failed to save the swapped face crop to {output_path}.")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Output file was not created correctly: {output_path}")
    return output_path


if __name__ == "__main__":
    raise SystemExit(main())
