import math
import unittest
from lhmpp_camera_framing import framing_transform, transform_intrinsics


class CameraFramingTests(unittest.TestCase):
    def test_projection_matches_crop_then_resize(self):
        transform = framing_transform(1280, 720, 420, [300, 120, 900, 720])
        focal, principal = transform_intrinsics([900, 850], [640, 360], transform)
        # Several 3D rays must land at the same pixels after crop/resize and
        # after projecting with the transformed camera.
        for x, y, z in [(0, 0, 3), (0.4, -0.2, 2), (-0.3, 0.8, 4)]:
            u, v = 900 * x / z + 640, 850 * y / z + 360
            self.assertAlmostEqual(focal[0] * x / z + principal[0],
                                   (u - 300) * transform['width'] / 600)
            self.assertAlmostEqual(focal[1] * y / z + principal[1], (v - 120) * 420 / 600)

    def test_full_frame_legacy_size(self):
        transform = framing_transform(1280, 720, 420)
        self.assertEqual((transform['width'], transform['height']), (742, 420))

    def test_crop_reduces_rendered_background(self):
        full = framing_transform(1280, 720, 420)
        crop = framing_transform(1280, 720, 420, [300, 120, 900, 720])
        self.assertLess(crop['width'], full['width'])
        self.assertEqual(crop['width'] % 14, 0)

    def test_invalid_boxes(self):
        for box in [[-1, 0, 20, 20], [0, 0, 1281, 720], [5, 5, 5, 20],
                    [0, 0, math.nan, 20], [0, 0, 20, math.inf]]:
            with self.subTest(box=box), self.assertRaises(ValueError):
                framing_transform(1280, 720, 420, box)

    def test_patch_rounding_preserves_body_proportions(self):
        transform = framing_transform(1280, 720, 420, [301, 121, 904, 714])
        focal, _ = transform_intrinsics([900, 850], [640, 360], transform)
        self.assertAlmostEqual(focal[0] / 900, focal[1] / 850)
        self.assertGreaterEqual(transform['pad_x'], 0)
        self.assertLess(transform['pad_x'], 7)


if __name__ == '__main__':
    unittest.main()
