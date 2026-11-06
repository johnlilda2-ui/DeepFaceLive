#!/usr/bin/env python3
"""Bounded CPU MediaPipe -> SMPL-X motion test using a fictional avatar clip."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

from avatar_body_retarget import MediaPipeBodyRetargeter, LANDMARK_INDICES
from headless_avatar_worker import validate_pose

MODEL_SHA256 = '59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a'


def infer(detector, image, timestamp, retargeter):
    begin = time.perf_counter()
    result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(image)), timestamp)
    tracking_ms = (time.perf_counter() - begin) * 1000
    if not result.pose_world_landmarks:
        return None, {'status': 'no_person', 'tracking_ms': tracking_ms}
    body = result.pose_world_landmarks[0]
    normalized = result.pose_landmarks[0]
    points = np.array([[p.x, p.y, p.z] for p in body])
    confidence = np.array([min(
        p.visibility if p.visibility is not None else 0,
        p.presence if p.presence is not None else 1) for p in normalized])
    begin = time.perf_counter()
    try:
        packet = retargeter.convert(points, confidence)
    except ValueError as error:
        return None, {'status': 'retarget_rejected', 'reason': str(error), 'tracking_ms': tracking_ms}
    retarget_ms = (time.perf_counter() - begin) * 1000
    details = {'status': 'ok' if packet is not None else 'low_confidence',
        'tracking_ms': tracking_ms, 'retarget_ms': retarget_ms,
        'minimum_required_confidence': float(confidence[list(LANDMARK_INDICES.values())].min()),
        'landmarks_image': [[p.x, p.y] for p in normalized],
        'landmarks_world': points.tolist(), 'confidence': confidence.tolist()}
    return packet, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'input-video', 'camera-template', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != MODEL_SHA256:
        raise RuntimeError('Unexpected tracker model checksum')
    if mp.__version__ != '0.10.35':
        raise RuntimeError('This test requires mediapipe 0.10.35')
    template = json.loads(args.camera_template.read_text())
    retargeter = MediaPipeBodyRetargeter(template)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model),
            delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    capture = cv2.VideoCapture(str(args.input_video))
    if not capture.isOpened():
        raise RuntimeError('Cannot decode input clip')
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or not 1 <= fps <= 60:
        raise RuntimeError('Unsupported input FPS')
    records, packets = [], []
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
        try:
            for index in range(18):
                ok, bgr = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                timestamp = int(round(index * 1000 / fps))
                packet, details = infer(detector, rgb, timestamp, retargeter)
                details.update(frame_id=index, timestamp_ms=timestamp)
                records.append(details)
                if packet is not None:
                    validate_pose(packet)
                    packets.append({'frame_id': index, 'timestamp_ms': timestamp, 'pose': packet,
                                    'landmarks_image': details['landmarks_image'],
                                    'confidence': details['confidence']})
                print('TRACK_FRAME', index, details['status'],
                      details.get('minimum_required_confidence'), flush=True)
            blank_statuses = []
            for i in range(3):
                blank = np.full_like(rgb, 255)
                packet, details = infer(detector, blank,
                    int(round((len(records) + i) * 1000 / fps)), retargeter)
                blank_statuses.append({'status': details['status'], 'packet_emitted': packet is not None})
        finally:
            capture.release()
    report = {'scope': 'offline_cpu_body_tracking', 'live_verified': False,
        'mediapipe': mp.__version__, 'python': sys.version.split()[0],
        'delegate': 'CPU', 'torch_imported': 'torch' in sys.modules,
        'model_sha256': MODEL_SHA256, 'input_video': str(args.input_video),
        'source_frames': len(records), 'accepted_frames': len(packets),
        'coverage': len(packets) / max(1, len(records)),
        'blank_frames': blank_statuses, 'body_only': True,
        'translation_mode': 'pelvis_locked', 'camera_alignment': 'pi around X',
        'minimum_coverage_gate': 0.8, 'records': records}
    warm = [r['tracking_ms'] + r.get('retarget_ms', 0) for r in records[1:]]
    report['warm_median_tracking_and_retarget_ms'] = float(np.median(warm)) if warm else None
    report['warm_p95_tracking_and_retarget_ms'] = float(np.percentile(warm, 95)) if warm else None
    report['passed'] = (len(records) == 18 and len(packets) >= 15
                        and not any(b['packet_emitted'] for b in blank_statuses))
    (output / 'motion.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in packets))
    (output / 'tracking-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('TRACKER_SUMMARY', json.dumps({k:v for k,v in report.items() if k!='records'}), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
