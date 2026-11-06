#!/usr/bin/env python3
"""Bounded opt-in head/hand refinement of validated body-tracking packets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ['MPLBACKEND'] = 'Agg'

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
from avatar_head_hands import NeutralHeadCalibration, HandRetargeter
from avatar_body_retarget import rotation_matrix
from headless_avatar_worker import validate_pose
from bounded_motion_test import validate_records, minimum_body_frames

HAND_SHA256 = 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1'
# MP point order: wrist, thumb, index, middle, ring, pinky.
SMPLX_HAND_JOINTS = {
    'left': [20,37,38,39,None,25,26,27,None,28,29,30,None,34,35,36,None,31,32,33,None],
    'right':[21,52,53,54,None,40,41,42,None,43,44,45,None,49,50,51,None,46,47,48,None]}
SMPLX_TIPS = {'left':{4:5361,8:4933,12:5058,16:5169,20:5286},
              'right':{4:8079,8:7669,12:7794,16:7905,20:8022}}


def rest_hands(model, manifest):
    entries = json.loads(manifest.read_text())
    expected = [x for x in entries if x['path']=='human_model_files/smplx/SMPLX_NEUTRAL.npz']
    if len(expected)!=1 or hashlib.sha256(model.read_bytes()).hexdigest()!=expected[0]['sha256']:
        raise ValueError('SMPL-X asset checksum not in the prepared manifest')
    with np.load(model,allow_pickle=False) as data:
        vertices = data['v_template']
        joints = data['J_regressor'] @ vertices
    if joints.shape!=(55,3):
        raise ValueError('Require the expected SMPL-X 55-joint asset')
    return {side:np.array([vertices[SMPLX_TIPS[side][i]] if j is None else joints[j]
                          for i,j in enumerate(order)])
            for side,order in SMPLX_HAND_JOINTS.items()}


def hand_roi(image, normalized, side):
    h,w = image.shape[:2]
    wrist,elbow,tips = (15,13,[17,19,21]) if side=='left' else (16,14,[18,20,22])
    points = np.asarray(normalized,float)*[w,h]
    if points.shape!=(33,2) or not np.isfinite(points).all():
        raise ValueError('Invalid body-image landmarks')
    center = points[wrist] + 0.6*(points[tips].mean(0)-points[wrist])
    size = min(max(80.,1.65*np.linalg.norm(points[wrist]-points[elbow])),max(w,h)*0.4)
    x0,y0=np.maximum(0,np.floor(center-size/2)).astype(int)
    x1,y1=np.minimum([w,h],np.ceil(center+size/2)).astype(int)
    if x1-x0<32 or y1-y0<32:
        raise ValueError('Hand ROI falls outside the image')
    return image[y0:y1,x0:x1].copy(), np.array([x0,y0]), points[wrist]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('smplx-model','asset-manifest','hand-model','body-report','motion-jsonl','driver-video','output-dir'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--neutral-frame',required=True,type=int,
                        help='Explicit known level/front-facing reference frame')
    parser.add_argument('--frames',type=int,default=18)
    parser.add_argument('--diagnostic',action='store_true',help='Allow failed body coverage; never report it as passing')
    args=parser.parse_args()
    if mp.__version__!='0.10.35' or hashlib.sha256(args.hand_model.read_bytes()).hexdigest()!=HAND_SHA256:
        raise ValueError('Unexpected hand runtime or model')
    body=json.loads(args.body_report.read_text())
    inputs=[json.loads(x) for x in args.motion_jsonl.read_text().splitlines() if x.strip()]
    validate_records(inputs,args.frames)
    if body['source_frames']!=args.frames or (not body['passed'] and not args.diagnostic):
        raise ValueError('Require matching source timeline and passed body coverage, or explicit diagnostic mode')
    records={x['frame_id']:x for x in body['records']}
    poses={x['frame_id']:x for x in inputs}
    if args.neutral_frame not in poses:
        raise ValueError('Neutral reference is not a validated body frame')
    rest=rest_hands(args.smplx_model,args.asset_manifest)
    hands={s:HandRetargeter(rest[s],s) for s in rest}
    head=NeutralHeadCalibration(); head.calibrate(poses[args.neutral_frame]['pose'])
    options=mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.hand_model),
            delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,num_hands=1,
        min_hand_detection_confidence=0.5,min_hand_presence_confidence=0.5)
    output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=False)
    result_records, diagnostics=[],[]
    cap=cv2.VideoCapture(str(args.driver_video))
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        try:
            for frame_id in range(args.frames):
                ok,bgr=cap.read()
                if not ok: raise ValueError('Driving clip ended early')
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                if frame_id not in poses:
                    diagnostics.append({'frame_id':frame_id,'status':'body_missing',
                        'hands':{s:{'status':'body_missing'} for s in hands}})
                    for tracker in hands.values(): tracker.reset()
                    continue
                original=poses[frame_id]
                packet=head.apply(original['pose'])
                detail={'frame_id':frame_id,'hands':{},
                    'head_before_degrees':float(np.degrees(np.linalg.norm(original['pose']['body_pose'][14]))),
                    'head_after_degrees':float(np.degrees(np.linalg.norm(packet['body_pose'][14])))}
                for side in ('left','right'):
                    status={'status':'missing'}
                    try:
                        roi,offset,expected_wrist=hand_roi(rgb,records[frame_id]['landmarks_image'],side)
                        result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                       data=np.ascontiguousarray(roi)))
                        if result.hand_world_landmarks:
                            observed=np.array([[p.x,p.y] for p in result.hand_landmarks[0]])
                            wrist=observed[0]*[roi.shape[1],roi.shape[0]]+offset
                            error=float(np.linalg.norm(wrist-expected_wrist))
                            if error>max(12.,0.35*max(roi.shape[:2])):
                                raise ValueError('Detected wrist does not match the body wrist')
                            points=np.array([[p.x,p.y,p.z] for p in result.hand_world_landmarks[0]])
                            packet=hands[side].apply(packet,points*[1,-1,-1])
                            status={'status':'tracked','wrist_error_pixels':error,
                                'landmarks_image':(observed*[roi.shape[1],roi.shape[0]]+offset).tolist(),
                                'landmarks_world':points.tolist()}
                    except ValueError as error:
                        status={'status':'rejected','reason':str(error)}
                    if status['status']!='tracked':
                        hands[side].reset()
                    detail['hands'][side]=status
                packet=validate_pose(packet)
                new=dict(original); new['pose']=packet
                result_records.append(new); diagnostics.append(detail)
                print('HEAD_HAND_FRAME',frame_id,
                      {s:detail['hands'][s]['status'] for s in hands},
                      round(detail['head_after_degrees'],2),flush=True)
        finally: cap.release()
    counts={s:sum(d['hands'][s]['status']=='tracked' for d in diagnostics) for s in hands}
    report={'scope':'offline_neutral_head_and_geometric_hands','live_verified':False,
        'neutral_reference_frame':args.neutral_frame,'frames':args.frames,'refined_frames':len(result_records),
        'body_coverage_passed':body['passed'],'diagnostic_mode':args.diagnostic,
        'hand_tracked_frames':counts,'hand_model_sha256':HAND_SHA256,
        'head_reference_degrees_before':float(np.degrees(np.linalg.norm(poses[args.neutral_frame]['pose']['body_pose'][14]))),
        'head_reference_degrees_after':next(d['head_after_degrees'] for d in diagnostics if d['frame_id']==args.neutral_frame),
        'head_after_max_degrees':max(d['head_after_degrees'] for d in diagnostics if 'head_after_degrees' in d),
        'finger_max_radians':{s:max(float(np.max(np.linalg.norm(r['pose'][key],axis=1)))
                                  for r in result_records)
                              for s,key in [('left','lhand_pose'),('right','rhand_pose')]},
        'minimum_hand_observations':int(np.ceil(2*args.frames/3)),
        'passed':body['passed'] and len(result_records)>=minimum_body_frames(args.frames)
                 and min(counts.values())>=int(np.ceil(2*args.frames/3))
                 and next(d['head_after_degrees'] for d in diagnostics if d['frame_id']==args.neutral_frame)<0.01,
        'limitations':['Explicit level neutral-head calibration required',
                       'Geometric hand approximation and smoothing; not ground truth',
                       'No lip sync or facial expression tracking']}
    (output/'motion.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in result_records))
    (output/'head-hand-diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    (output/'head-hand-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('HEAD_HAND_SUMMARY',json.dumps(report),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
