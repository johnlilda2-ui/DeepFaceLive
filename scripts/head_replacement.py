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
    _, top, _, head_height = cv2.boundingRect((source_mask > .5).astype(np.uint8))
    # Feather the lower seam without softening hair detail.
    lower = np.clip((np.arange(size) - top - .68 * head_height) / (.2 * head_height), 0, 1)
    lower = lower.astype(np.float32)[:, None, None]
    premultiplied = np.dstack((generated.astype(np.float32) * source_mask[..., None], source_mask))
    softened = cv2.GaussianBlur(premultiplied, (0, 0), max(1, size / 115))
    boundary = np.clip((1 - softened[..., 3:4]) * 5, 0, 1)
    premultiplied += lower * boundary * (softened - premultiplied)
    warped = cv2.warpAffine(premultiplied, generated_matrix, (width, height))
    replacement, alpha = warped[..., :3], warped[..., 3]
    if not np.any(alpha > .5):
        return None
    old_h, old_w = driver_mask.shape
    driver_matrix = inverse.to_exact_mat(old_w, old_h, width, height)
    # The head mask can underestimate the ears. Grow it laterally above
    # the jaw so cleanup can cover their edges while protecting the neck.
    _, old_top, _, old_height = cv2.boundingRect((driver_mask > .5).astype(np.uint8))
    upper = old_top + round(.76 * old_height)
    radius = max(1, old_w // 32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 3))
    cleanup = driver_mask.copy()
    cleanup[:upper] = cv2.dilate(driver_mask, kernel)[:upper]
    old_head = cv2.warpAffine(cleanup, driver_matrix, (width, height))
    old_mask = (old_head > .35).astype(np.uint8)
    clean = frame
    if np.any(old_mask & (alpha < .98)):
        # Mask the covered part as well, so inpainting cannot pull old skin
        # or hair from under the replacement into its exposed border.
        clean = frame.copy()
        x, y, w, h = cv2.boundingRect(old_mask)
        area = np.s_[max(0, y - 8):y + h + 8, max(0, x - 8):x + w + 8]
        clean[area] = cv2.inpaint(frame[area], old_mask[area], 3, cv2.INPAINT_TELEA)
    return np.clip(np.rint(clean * (1 - alpha[..., None]) + replacement), 0, 255).astype(np.uint8)
