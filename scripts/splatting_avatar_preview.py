"""Fixed-camera body tracking and room compositing for the sample preview."""
from pathlib import Path

import cv2
import numpy as np

from avatar_body_retarget import BODY_JOINT_NAMES, LANDMARK_INDICES, MediaPipeBodyRetargeter, PoseCapError
from headless_avatar_worker import validate_pose
from headless_lhmpp_stream_worker import HeadlessLHMStreamWorker


def portrait_crop(frame, size):
    width, height = size
    source_height, source_width = frame.shape[:2]
    if source_width * height > source_height * width:
        crop_width = max(1, round(source_height * width / height))
        left = (source_width - crop_width) // 2
        frame = frame[:, left:left + crop_width]
    else:
        crop_height = max(1, round(source_width * height / width))
        top = (source_height - crop_height) // 2
        frame = frame[top:top + crop_height]
    return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)


class SplattingAvatarTracker:
    def __init__(self, config):
        import mediapipe as mp

        self.mp = mp
        self.detector = None
        self.template = validate_pose(config['initial_pose'])
        self.size = tuple(map(int, self.template['img_size_wh']))
        if self.size != (320, 576):
            raise ValueError('The SplattingAvatar preview uses a 320 × 576 camera.')
        model = Path(config['pose_model'])
        if not model.is_file():
            raise ValueError('Supply a local MediaPipe Pose Landmarker model as pose_model.')
        self.options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            output_segmentation_masks=True,
        )
        self.reset()

    def reset(self):
        self.close()
        self.retarget = MediaPipeBodyRetargeter(self.template)
        self.detector = self.mp.tasks.vision.PoseLandmarker.create_from_options(self.options)
        self.mask = self.position = None
        self.last_timestamp = -1

    def close(self):
        if self.detector is not None:
            self.detector.close()
            self.detector = None

    def convert(self, bgr, timestamp_ms):
        timestamp = max(self.last_timestamp + 1, int(timestamp_ms))
        self.last_timestamp = timestamp
        bgr = portrait_crop(bgr, self.size)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self.detector.detect_for_video(self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB, data=rgb), timestamp)
        self.mask = None
        if not (result.pose_landmarks and result.pose_world_landmarks and result.segmentation_masks):
            self.position = None
            return None, {'body': False}
        points = result.pose_landmarks[0]
        world = np.array([[p.x, p.y, p.z] for p in result.pose_world_landmarks[0]])
        confidence = np.array([min(p.visibility or 0, p.presence if p.presence is not None else 1)
                               for p in points])
        weak_feet = []
        for side in ('left', 'right'):
            ankle, heel, toe = (LANDMARK_INDICES[f'{side}_{joint}']
                                 for joint in ('ankle', 'heel', 'foot_index'))
            if confidence[ankle] >= .25 and min(confidence[[ankle, heel, toe]]) < .5:
                # Retain knee motion from the observed ankle. Ignore uncertain foot rotations.
                weak_feet.append(side)
                confidence[[ankle, heel, toe]] = .5
        try:
            pose = self.retarget.convert(world, confidence)
        except PoseCapError:
            pose = None
        if pose is None:
            self.position = None
            return None, {'body': False}
        for side in weak_feet:
            for joint in ('ankle', 'foot'):
                pose['body_pose'][BODY_JOINT_NAMES.index(f'{side}_{joint}')] = [0, 0, 0]
        width, height = self.size
        pixel = np.array([[p.x * width, p.y * height] for p in points])
        hips = pixel[[23, 24]].mean(0)
        shoulders = pixel[[11, 12]].mean(0)
        torso_pixels = np.linalg.norm(shoulders - hips)
        torso_metres = np.linalg.norm(world[[11, 12]].mean(0) - world[[23, 24]].mean(0))
        if torso_pixels < 20 or not .15 < torso_metres < .9:
            self.position = None
            return None, {'body': False}
        fx, fy = pose['focal']
        cx, cy = pose['princpt']
        depth = np.clip(fy * torso_metres / torso_pixels, .8, 8)
        position = np.array([(hips[0] - cx) * depth / fx, (hips[1] - cy) * depth / fy, depth])
        self.position = position if self.position is None else .55 * self.position + .45 * position
        pose['trans'] = self.position.tolist()
        self.mask = result.segmentation_masks[0].numpy_view().squeeze().copy()
        return pose, {'body': True, 'feet_neutralized': weak_feet}


class SplattingAvatarPreviewWorker(HeadlessLHMStreamWorker):
    engine = 'splattingavatar'

    def __init__(self, tracker, renderer, room):
        super().__init__(tracker, renderer)
        self.room = portrait_crop(room, tracker.size)
        self.margin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))

    def output_frame(self, image, rgba):
        width, height = self.tracker.size
        camera = portrait_crop(image, self.tracker.size)
        mask = cv2.resize(self.tracker.mask, (width, height))
        if not np.isfinite(mask).all() or not np.any(mask > .2):
            return np.zeros_like(camera), 'avatar_no_body'
        remove = cv2.dilate((mask > .2).astype(np.uint8), self.margin).astype(np.float32)
        remove = np.maximum(remove, cv2.GaussianBlur(remove, (0, 0), 5))[..., None]
        background = camera * (1 - remove) + self.room * remove
        opacity = rgba[..., 3:4].astype(np.float32) / 255
        foreground_bgr = rgba[..., 2::-1].astype(np.float32)
        result = foreground_bgr + (1 - opacity) * background
        return np.clip(np.rint(result), 0, 255).astype(np.uint8), 'avatar'
