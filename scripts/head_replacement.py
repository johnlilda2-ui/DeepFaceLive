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

    def regions(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = self.segmenter.segment(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        masks = result.confidence_masks
        if masks is None or len(masks) != 6:
            raise RuntimeError('The head segmenter returned an unexpected class map.')
        # Hair, exposed body skin, face skin, and accessories respectively.
        return tuple(masks[i].numpy_view().reshape(bgr.shape[:2]).copy()
                     for i in (1, 2, 3, 5))

    def close(self):
        self.segmenter.close()


def opacity(probability):
    return np.clip((probability - .30) / .48, 0, 1)


def composite_head(frame, driver_crop, generated, transform, segmenter):
    """Attach the rendered head and upper neck to the original camera frame."""
    height, width = frame.shape[:2]
    size = generated.shape[0]
    if generated.shape != (size, size, 3) or generated.dtype != np.uint8:
        raise ValueError('Invalid head decoder output.')

    hair, body_skin, face, _ = segmenter.regions(generated)
    _, _, driver_face, _ = segmenter.regions(driver_crop)
    source_box = cv2.boundingRect((face > .65).astype(np.uint8))
    driver_box = cv2.boundingRect((driver_face > .65).astype(np.uint8))
    sx, sy, sw, sh = source_box
    dx, dy, dw, dh = driver_box
    if min(sw, sh, dw, dh) < 4:
        return None

    # Both crops share the same aligned coordinates. Anchor the face there so
    # motion inside the generator cannot slide the whole head off the neck.
    crop_size = driver_crop.shape[0]
    scale = float(np.clip(dh * size / (sh * crop_size), .75, 1.25))
    face_center = np.array([sx + sw / 2, sy + sh / 2], dtype=np.float32)
    driver_center = np.array([dx + dw / 2, dy + dh / 2], dtype=np.float32)
    driver_center *= size / crop_size
    translation = driver_center - scale * face_center
    fit = np.array([[scale, 0, translation[0]], [0, scale, translation[1]], [0, 0, 1]],
                   dtype=np.float32)
    matrix = transform.invert().to_exact_mat(size, size, width, height) @ fit

    y, x = np.ogrid[:size, :size]
    # Exposed skin also includes arms and chest. Use just the area directly
    # below the jaw and fade its lower edge into the camera's neck.
    jaw = sy + sh
    neck_fade = np.clip((jaw + .18 * sh - y) / (.18 * sh), 0, 1)
    neck_area = (y > sy + .65 * sh) & (abs(x - face_center[0]) < .58 * sw)
    neck = opacity(body_skin) * neck_fade * neck_area
    alpha = np.maximum(opacity(hair + face), neck).astype(np.float32)
    premultiplied = np.dstack((generated.astype(np.float32) * alpha[..., None],
                               alpha, neck.astype(np.float32)))
    warped = cv2.warpAffine(premultiplied, matrix, (width, height))
    replacement, alpha = warped[..., :3], warped[..., 3]
    if not np.any(alpha > .5):
        return None

    rendered_neck = warped[..., 4]

    # Segment the whole camera frame to cover hair below the aligned crop.
    old_hair, old_skin, old_face, accessories = segmenter.regions(frame)
    old_head = (old_hair + old_face > .30).astype(np.uint8)
    count, labels = cv2.connectedComponents(old_head)
    driver_matrix = transform.invert().to_exact_mat(crop_size, crop_size, width, height)
    face_in_frame = cv2.warpAffine(driver_face, driver_matrix, (width, height))
    overlap = np.bincount(labels[face_in_frame > .65], minlength=count)
    overlap[0] = 0
    if not np.any(overlap):
        return None
    old_head = (labels == overlap.argmax()).astype(np.uint8)
    _, _, old_width, _ = cv2.boundingRect(old_head)
    radius = max(1, round(old_width * .045))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    near_head = cv2.dilate(old_head, kernel, iterations=2)
    old_head |= ((accessories > .4) & (near_head != 0)).astype(np.uint8)
    old_mask = cv2.dilate(old_head, kernel)
    old_mask[(old_skin > .65) & (old_head == 0)] = 0
    rows, columns = np.ogrid[:crop_size, :crop_size]
    fade = np.clip((dy + 1.35 * dh - rows) / (.26 * dh), 0, 1)
    fade *= rows > dy + .67 * dh
    neck_zone = abs(columns - (dx + dw / 2)) < .8 * dw
    under_jaw = cv2.warpAffine(
        (fade * neck_zone).astype(np.float32), driver_matrix, (width, height))

    clean = frame.astype(np.float32)
    cleanup_strength = np.zeros((height, width), dtype=np.float32)
    if np.any(old_mask & (alpha < .98)):
        # Exclude nearby neck pixels from the fill samples. Feather only
        # the perimeter of the expanded mask, outside the original head.
        fill_mask = old_mask | ((old_skin > .30) & (under_jaw > 0)).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(fill_mask)
        area = np.s_[max(0, y - 8):y + h + 8, max(0, x - 8):x + w + 8]
        filled = cv2.inpaint(frame[area], fill_mask[area], 3, cv2.INPAINT_NS)
        edge = cv2.distanceTransform(old_mask[area], cv2.DIST_L2, 3)
        cleanup_strength[area] = np.clip(edge / max(1, radius * .55), 0, 1)
        clean[area] += cleanup_strength[area][..., None] * (filled - clean[area])

    # Match the skin on either side of the seam using pixels where both
    # necks are visible. Taper the adjustment below the jaw and into the
    # skin mask so shoulders, clothes and background retain their color.
    matching_skin = (old_skin > .8) & (rendered_neck > .55) & (under_jaw > .6)
    if np.count_nonzero(matching_skin) >= 16:
        rendered_color = replacement[matching_skin] / alpha[matching_skin, None]
        shift = np.median(rendered_color - frame[matching_skin], axis=0)
        strength = under_jaw * opacity(old_skin) * (1 - cleanup_strength)
        clean += strength[..., None] * np.clip(shift, -100, 100)
    return np.clip(np.rint(clean * (1 - alpha[..., None]) + replacement), 0, 255).astype(np.uint8)
