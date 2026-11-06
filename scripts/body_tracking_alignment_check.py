#!/usr/bin/env python3
"""CPU image-space consistency screen for the tracked-avatar replay.

This reuses the pose detector, so it is not independent ground-truth accuracy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ['MPLBACKEND'] = 'Agg'

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageOps

BONES = [(11,13),(13,15),(12,14),(14,16),(23,25),(25,27),(24,26),(26,28)]


def angle(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1 or nb < 1:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model','motion-jsonl','driver-video','rendered-video','output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    from body_tracking_smoke_test import MODEL_SHA256
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != MODEL_SHA256:
        raise RuntimeError('Unexpected detector checksum')
    records = {r['frame_id']:r for r in
               [json.loads(line) for line in args.motion_jsonl.read_text().splitlines() if line.strip()]}
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model),
            delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_poses=1)
    a, b = cv2.VideoCapture(str(args.driver_video)), cv2.VideoCapture(str(args.rendered_video))
    paired, scores = [], []
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
            for i in range(18):
                ok1, source = a.read(); ok2, rendered = b.read()
                if not ok1 or not ok2: break
                rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
                result = detector.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb)),
                    int(round(i*1000/6)))
                if i not in records or not result.pose_landmarks:
                    continue
                normalized = result.pose_landmarks[0]
                dst = np.array([[p.x*rendered.shape[1],p.y*rendered.shape[0]] for p in normalized])
                src = np.array(records[i]['landmarks_image']) * [source.shape[1],source.shape[0]]
                confidence = [min(p.visibility if p.visibility is not None else 0,
                                  p.presence if p.presence is not None else 1) for p in normalized]
                errors = []
                for joint_a, joint_b in BONES:
                    if min(confidence[joint_a],confidence[joint_b]) < 0.5:
                        continue
                    if not np.isfinite(dst[[joint_a,joint_b]]).all():
                        continue
                    error = angle(src[joint_b]-src[joint_a],dst[joint_b]-dst[joint_a])
                    if error is not None: errors.append(error)
                if errors:
                    scores.append({'frame_id':i,'bone_errors_degrees':errors})
                    paired.append((i,cv2.cvtColor(source,cv2.COLOR_BGR2RGB),rgb,src,dst))
    finally:
        a.release(); b.release()
    values = [e for s in scores for e in s['bone_errors_degrees']]
    report = {'scope':'offline_pose_image_consistency','independent_ground_truth':False,
        'live_verified':False,'matched_frames':len(scores),'bone_samples':len(values),
        'mean_angle_degrees':float(np.mean(values)) if values else None,
        'p90_angle_degrees':float(np.percentile(values,90)) if values else None,
        'limits':{'minimum_matched_frames':12,'minimum_bone_samples':80,
                  'maximum_mean_degrees':35,'maximum_p90_degrees':60},'scores':scores}
    report['passed'] = (len(scores)>=12 and len(values)>=80 and
                       report['mean_angle_degrees']<=35 and report['p90_angle_degrees']<=60)
    output = args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=True)
    (output/'alignment-report.json').write_text(json.dumps(report,indent=2)+'\n')
    if paired:
        board = Image.new('RGB',(950,990),'white'); draw=ImageDraw.Draw(board)
        draw.text((12,10),'Driving body (left) | Retargeted avatar (right) | CPU skeleton overlay',fill='black')
        for row, sample in enumerate([paired[0],paired[len(paired)//2],paired[-1]]):
            index, source, rendered, src, dst = sample
            for frame, points, left, width in [(source,src,0,300),(rendered,dst,310,630)]:
                image = Image.fromarray(frame); pen = ImageDraw.Draw(image)
                for j,(k,l) in enumerate(BONES):
                    if np.isfinite(points[[k,l]]).all():
                        pen.line([tuple(points[k]),tuple(points[l])],
                                 fill='red' if j in (0,1,4,5) else 'blue',width=3)
                tile=ImageOps.contain(image,(width,280))
                board.paste(tile,(left+(width-tile.width)//2,45+row*310))
            draw.text((12,330+row*310),f'Frame {index+1}',fill='black')
        board.save(output/'alignment-comparison.png')
    print(json.dumps(report,indent=2),flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
