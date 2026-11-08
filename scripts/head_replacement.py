"""Segment and composite a rendered head in the original webcam frame."""
import hashlib
from pathlib import Path

import cv2
import numpy as np


class HeadSegmenter:
    """MediaPipe SelfieMulticlass float32 v1, using hair and face classes."""
    sha256 = 'c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0'

    def __init__(self, model_path):
        import mediapipe as mp

        model = Path(model_path)
        if not model.is_file() or hashlib.sha256(model.read_bytes()).hexdigest() != self.sha256:
            raise ValueError('Use the official SelfieMulticlass float32 v1 model for head replacement.')
        self.mp = mp
        vision = mp.tasks.vision
        options = vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            output_confidence_masks=True,
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

    def mask(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self.segmenter.segment(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        masks = result.confidence_masks
        if masks is None or len(masks) != 6:
            raise RuntimeError('The head segmenter returned an unexpected class map.')
        probability = masks[1].numpy_view() + masks[3].numpy_view()
        return np.clip((probability.reshape(bgr.shape[:2]) - .30) / .48, 0, 1)

    def close(self):
        self.segmenter.close()


def composite_head(frame, driver_crop, generated, transform, segmenter):
    """Blend a cut-out replacement head. Fill exposed old head pixels only."""
    height, width = frame.shape[:2]
    size = generated.shape[0]
    if generated.shape != (size, size, 3) or generated.dtype != np.uint8:
        raise ValueError('Invalid head decoder output.')
    source_mask = segmenter.mask(generated)
    driver_mask = segmenter.mask(driver_crop)
    if not np.any(source_mask > .5) or not np.any(driver_mask > .5):
        return None

    inverse = transform.invert()
    generated_matrix = inverse.to_exact_mat(size, size, width, height)
    alpha = cv2.warpAffine(source_mask, generated_matrix, (width, height))
    if not np.any(alpha > .5):
        return None
    replacement = cv2.warpAffine(generated.astype(np.float32) * source_mask[..., None],
                                 generated_matrix, (width, height))
    old_h, old_w = driver_mask.shape
    driver_matrix = inverse.to_exact_mat(old_w, old_h, width, height)
    old_head = cv2.warpAffine(driver_mask, driver_matrix, (width, height))
    exposed = (old_head > .35) & (alpha < .98)
    clean = frame
    if exposed.any():
        clean = frame.copy()
        x, y, w, h = cv2.boundingRect(exposed.astype(np.uint8))
        area = np.s_[max(0, y - 8):y + h + 8, max(0, x - 8):x + w + 8]
        clean[area] = cv2.inpaint(frame[area], exposed[area].astype(np.uint8), 3, cv2.INPAINT_TELEA)
    return np.clip(np.rint(clean * (1 - alpha[..., None]) + replacement), 0, 255).astype(np.uint8)
