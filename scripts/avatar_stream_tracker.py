"""CPU-only body/head/hands/jaw tracker for the experimental LHM++ stream."""
import hashlib
import os
from pathlib import Path
os.environ['MPLBACKEND']='Agg'
import cv2
import mediapipe as mp
import numpy as np
from avatar_body_retarget import MediaPipeBodyRetargeter
from avatar_head_hands import NeutralHeadCalibration,HandRetargeter
from avatar_hand_guard import HandObservationGuard
from avatar_mouth_motion import JawOpening
from avatar_lip_expression import LipExpressions
from head_hand_tracking_test import rest_hands,hand_roi
from mouth_tracking_test import head_crop
from headless_avatar_worker import validate_pose

MODEL_HASHES=dict(pose_model='59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a',
    hand_model='fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1',
    face_model='64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff')


class MediaPipeAvatarTracker:
    def __init__(self,config):
        if mp.__version__!='0.10.35':raise ValueError('Require MediaPipe 0.10.35')
        self.config=config
        self.detectors=[]
        for name,digest in MODEL_HASHES.items():
            if hashlib.sha256(Path(config[name]).read_bytes()).hexdigest()!=digest:
                raise ValueError('Tracker model checksum mismatch: '+name)
        self.rest=rest_hands(Path(config['smplx_model']),Path(config['manifest']))
        self.reset()

    def reset(self):
        self.close()
        self.body=MediaPipeBodyRetargeter(self.config['initial_pose'])
        self.head=NeutralHeadCalibration()
        self.jaw=JawOpening()
        self.hands={s:HandRetargeter(self.rest[s],s) for s in self.rest}
        self.guards={s:HandObservationGuard() for s in self.rest}
        self.lips=LipExpressions.load(self.config['lip_map']) if self.config.get('lip_map') else None
        self.last_timestamp=-1
        base=lambda key:mp.tasks.BaseOptions(model_asset_path=str(self.config[key]),
                                              delegate=mp.tasks.BaseOptions.Delegate.CPU)
        vision=mp.tasks.vision
        try:
            self.pose=vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=base('pose_model'),running_mode=vision.RunningMode.VIDEO,num_poses=1,
                min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,min_tracking_confidence=.5))
            self.detectors.append(self.pose)
            self.face=vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=base('face_model'),running_mode=vision.RunningMode.VIDEO,num_faces=1,
                output_face_blendshapes=True,min_face_detection_confidence=.5,
                min_face_presence_confidence=.5,min_tracking_confidence=.5))
            self.detectors.append(self.face)
            self.hand=vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=base('hand_model'),running_mode=vision.RunningMode.IMAGE,num_hands=1,
                min_hand_detection_confidence=.5,min_hand_presence_confidence=.5))
            self.detectors.append(self.hand)
        except BaseException:
            self.close();raise

    @staticmethod
    def image(rgb):
        return mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb))

    def convert(self,bgr,timestamp_ms):
        timestamp=int(round(timestamp_ms))
        if timestamp<0 or timestamp<=self.last_timestamp:
            raise ValueError('Tracker timestamps must increase')
        self.last_timestamp=timestamp
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        result=self.pose.detect_for_video(self.image(rgb),timestamp)
        detail=dict(body=False,face=False,hands={})
        packet=None
        if result.pose_world_landmarks:
            points=np.array([[p.x,p.y,p.z] for p in result.pose_world_landmarks[0]])
            image_points=np.array([[p.x,p.y] for p in result.pose_landmarks[0]])
            confidence=np.array([min(p.visibility or 0.,p.presence if p.presence is not None else 1.)
                                 for p in result.pose_landmarks[0]])
            packet=self.body.convert(points,confidence)
        if packet is None:
            for hand in self.hands.values():hand.reset()
            for guard in self.guards.values():guard.reset()
            self.jaw=JawOpening()
            return None,detail
        detail['body']=True
        if self.config.get('calibrate_head_on_start',False):
            # Explicit opt-in: the operator must begin level and facing forward.
            if self.head.reference is None:self.head.calibrate(packet)
            packet=self.head.apply(packet)
        scores={}
        try:
            face=self.face.detect_for_video(self.image(head_crop(rgb,image_points)),timestamp)
            if face.face_blendshapes:
                scores={c.category_name:float(c.score) for c in face.face_blendshapes[0]}
        except ValueError:pass
        detail['face']='jawOpen' in scores
        packet=self.jaw.apply(packet,scores['jawOpen']) if detail['face'] else self.jaw.missing(packet)
        if self.lips is not None:packet=self.lips.apply(packet,scores,timestamp)
        for side in ('left','right'):
            status='missing'
            try:
                roi,offset,wrist=hand_roi(rgb,image_points,side)
                hand=self.hand.detect(self.image(roi))
                if hand.hand_world_landmarks:
                    observed=np.array([[p.x,p.y] for p in hand.hand_landmarks[0]])
                    wrist_error=np.linalg.norm(observed[0]*[roi.shape[1],roi.shape[0]]+offset-wrist)
                    if wrist_error>max(12.,.35*max(roi.shape[:2])):
                        raise ValueError('Hand does not match body wrist')
                    points=np.array([[p.x,p.y,p.z] for p in hand.hand_world_landmarks[0]])*[1,-1,-1]
                    if self.config.get('guard_hands',False):
                        allowed,status=self.guards[side].check(points,float(timestamp))
                        if not allowed:raise ValueError(status)
                    packet=self.hands[side].apply(packet,points)
                    status='tracked'
            except ValueError as error:status='rejected: '+str(error)
            if status!='tracked':packet=self.hands[side].relax(packet)
            detail['hands'][side]=status
        return validate_pose(packet),detail

    def close(self):
        for detector in self.detectors:detector.close()
        self.detectors=[]
