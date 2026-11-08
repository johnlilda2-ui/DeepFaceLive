import unittest

import numpy as np

from splatting_avatar_preview import SplattingAvatarPreviewWorker, portrait_crop


class Tracker:
    size = (320, 576)

    def __init__(self):
        self.mask = np.zeros((576, 320), np.float32)
        self.mask[140:500, 110:220] = 1
        self.missing = False

    def convert(self, image, timestamp_ms):
        return (None if self.missing else {}), {'body': not self.missing}

    def close(self):
        pass


class Renderer:
    shape = (576, 320, 4)

    def render(self, frame_id, timestamp_ms, pose):
        rgba = np.zeros(self.shape, np.uint8)
        rgba[160:480, 140:180] = [100, 0, 0, 128]
        return rgba

    def close(self):
        pass


class SplattingAvatarPreviewTests(unittest.TestCase):
    def test_body_is_replaced_and_uncovered_room_stays_live(self):
        tracker = Tracker()
        room = np.full((576, 320, 3), 80, np.uint8)
        frame = np.full_like(room, 20)
        frame[140:500, 110:220] = 255
        with self.subTest('premultiplied overlay and fixed room plate'):
            worker = SplattingAvatarPreviewWorker(tracker, Renderer(), room)
            self.addCleanup(worker.close)
            output, status = worker.process_video(frame, 0)
            self.assertEqual(status, 'avatar')
            np.testing.assert_array_equal(output[0, 0], [20, 20, 20])
            np.testing.assert_array_equal(output[200, 120], [80, 80, 80])
            np.testing.assert_array_equal(output[200, 160], [40, 40, 140])
        with self.subTest('tracking loss returns a blank RGB frame'):
            tracker.missing = True
            output, status = worker.process_video(frame, 40)
            self.assertEqual(status, 'avatar_no_body')
            self.assertEqual(output.shape, frame.shape)
            self.assertFalse(output.any())

    def test_crop_preserves_aspect_ratio(self):
        frame = np.full((576, 640, 3), 25, np.uint8)
        frame[:, 160:480] = 100
        np.testing.assert_array_equal(portrait_crop(frame, (320, 576)),
                                      np.full((576, 320, 3), 100, np.uint8))
