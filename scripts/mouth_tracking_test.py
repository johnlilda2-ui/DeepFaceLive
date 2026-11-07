"""Offline face-score to jaw-motion check; leaves the baseline JSONL intact."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['MPLBACKEND'] = 'Agg'
import cv2
import mediapipe as mp
import numpy as np
from avatar_mouth_motion import JawOpening
from headless_avatar_worker import validate_pose


def head_crop(rgb, landmarks):
    """Use already tracked body head points; a crop does not add detail."""
    points = np.asarray(landmarks, dtype=float)
    if points.shape != (33, 2) or not np.isfinite(points[:11]).all():
        raise ValueError('Invalid body head landmarks')
    h, w = rgb.shape[:2]
    points = points[:11]*[w, h]
    lo, hi = points.min(0), points.max(0)
    center = (lo+hi)/2
    extent = max(float((hi-lo).max())*2.0, 64.)
    x0,y0 = np.maximum(0, np.floor(center-extent/2)).astype(int)
    x1,y1 = np.minimum([w,h], np.ceil(center+extent/2)).astype(int)
    if min(x1-x0,y1-y0) < 32:
        raise ValueError('Head crop too small')
    return cv2.resize(rgb[y0:y1,x0:x1], (512,512), interpolation=cv2.INTER_LINEAR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'video', 'motion-jsonl', 'output-dir'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--model-sha256', required=True)
    parser.add_argument('--lip-map', type=Path, help='Opt-in private fitted lip-expression map; never guesses coefficient indices')
    args = parser.parse_args()
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != args.model_sha256:
        raise ValueError('Face model checksum mismatch')
    if args.motion_jsonl.stat().st_size > 180*16*1024:
        raise ValueError('Motion input exceeds test bound')
    records = [json.loads(s) for s in args.motion_jsonl.read_text().splitlines() if s.strip()]
    if len(records) != 180 or [r['frame_id'] for r in records] != list(range(180)):
        raise ValueError('Require the existing complete 180-frame sequence')
    for record in records:
        validate_pose(record['pose'])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model),
                                          delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True,
        min_face_detection_confidence=.5, min_face_presence_confidence=.5,
        min_tracking_confidence=.5)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError('Cannot decode source')
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid source FPS')
    jaw = JawOpening()
    lips = None
    if args.lip_map is not None:
        from avatar_lip_expression import LipExpressions
        lips = LipExpressions.load(args.lip_map)
    refined, diagnostics = [], []
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as tracker:
        try:
            for index, record in enumerate(records):
                timestamp = int(round(index*1000/fps))
                if abs(timestamp-record['timestamp_ms']) > 2:
                    raise ValueError('Source and motion timestamps disagree')
                ok, frame = capture.read()
                if not ok:
                    raise ValueError('Source clip ended early')
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = head_crop(rgb, record['landmarks_image'])
                result = tracker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
                scores = {c.category_name:float(c.score) for c in result.face_blendshapes[0]} if result.face_blendshapes else {}
                score = scores.get('jawOpen')
                pose = jaw.missing(record['pose']) if score is None else jaw.apply(record['pose'], score)
                if lips is not None:
                    pose = lips.apply(pose,scores,float(timestamp))
                refined.append(dict(record, pose=validate_pose(pose)))
                diagnostics.append(dict(frame_id=index, detected=score is not None,
                                        jaw_open_score=score, jaw_radians=pose['jaw_pose'][0]))
        finally:
            capture.release()
    observed = [r['jaw_open_score'] for r in diagnostics if r['detected']]
    report = dict(scope=('offline_video_driven_jaw_opening' if lips is None else 'offline_video_driven_jaw_and_fitted_lips'), live_verified=False,
                  audio_lip_sync_verified=False, fine_lip_shapes_verified=False,
                  frames=len(refined), tracked_frames=len(observed), mediapipe=mp.__version__,
                  model_sha256=args.model_sha256,
                  input_mode='body-landmark-guided head crop, resized to 512 square',
                  jaw_score_range=[min(observed), max(observed)] if observed else None,
                  jaw_radians_range=[min(r['jaw_radians'] for r in diagnostics), max(r['jaw_radians'] for r in diagnostics)],
                  tracking_passed=len(observed) >= 144,
                  visible_motion_excitation=len(observed)>0 and max(observed)-min(observed) >= .1,
                  visual_quality_verified=False,
                  fitted_lip_controls_enabled=lips is not None,
                  expression_max_norm=float(max(np.linalg.norm(r['pose']['expr']) for r in refined)),
                  limitations=['jawOpen to jaw X is an approximate bounded mapping',
                               ('No ARKit-to-FLAME expression basis conversion' if lips is None else
                                'Fitted lip controls have residual error; rejected controls stay disabled'),
                               'Missing faces ease mouth toward closed; no identity matching'])
    (output/'motion.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in refined))
    (output/'mouth-diagnostics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    (output/'mouth-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('MOUTH_TRACKING_REPORT',json.dumps(report),flush=True)
    return 0 if report['tracking_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

