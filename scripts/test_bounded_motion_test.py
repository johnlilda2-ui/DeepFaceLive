import unittest
from bounded_motion_test import validate_sampling, minimum_body_frames, validate_records


class TimelineTests(unittest.TestCase):
    def test_existing_and_longer_coverage_gates(self):
        self.assertEqual(minimum_body_frames(18),15)
        self.assertEqual(minimum_body_frames(180),144)

    def test_sparse_motion_keeps_source_ids(self):
        validate_records([{'frame_id':2,'timestamp_ms':167},
                          {'frame_id':179,'timestamp_ms':14917}],180)

    def test_duplicate_or_reordered_frames_and_times_are_rejected(self):
        for bad in [
            [{'frame_id':1,'timestamp_ms':1},{'frame_id':1,'timestamp_ms':2}],
            [{'frame_id':2,'timestamp_ms':1},{'frame_id':1,'timestamp_ms':2}],
            [{'frame_id':1,'timestamp_ms':2},{'frame_id':2,'timestamp_ms':2}],
            [{'frame_id':1,'timestamp_ms':2},{'frame_id':2,'timestamp_ms':1}]]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                validate_records(bad,18)

    def test_out_of_bounds_types_and_frame_budgets(self):
        for frame_id in (-1,18,True,1.0):
            with self.subTest(frame_id=frame_id),self.assertRaises(ValueError):
                validate_records([{'frame_id':frame_id,'timestamp_ms':0},
                                  {'frame_id':17,'timestamp_ms':1}],18)
        for count,fps in [(181,12),(0,6),(18,0),(18,31),(True,6)]:
            with self.subTest(count=count,fps=fps),self.assertRaises(ValueError):
                validate_sampling(count,fps)

    def test_noninteger_and_nonfinite_timestamps(self):
        for timestamp in (True,1.5,float('nan'),2**53):
            with self.subTest(timestamp=timestamp),self.assertRaises(ValueError):
                validate_records([{'frame_id':0,'timestamp_ms':0},
                                  {'frame_id':1,'timestamp_ms':timestamp}],18)


if __name__=='__main__':
    unittest.main()
