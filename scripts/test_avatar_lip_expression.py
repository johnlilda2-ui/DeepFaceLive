import copy
import unittest
import numpy as np
from avatar_lip_expression import fit_lip_basis, LipExpressions


class LipBasisTests(unittest.TestCase):
    def test_unrepresentable_control_rejected(self):
        dirs=np.zeros((40,3,100));dirs.reshape(-1,100)[:100]=np.eye(100)
        targets=np.zeros((2,40,3));targets.reshape(2,-1)[0,0]=1
        targets.reshape(2,-1)[1,110]=1
        matrix,meta=fit_lip_basis(dirs,targets,['mouthPucker','mouthFunnel'])
        self.assertEqual(meta['accepted'],[True,False])
        self.assertTrue(np.all(matrix[:,1]==0))

    def test_named_mapping_preserves_other_motion_and_state_on_invalid(self):
        matrix=np.zeros((100,1));matrix[7,0]=2
        tracker=LipExpressions(matrix,dict(names=['mouthPucker'],accepted=[True]))
        packet=dict(jaw_pose=[.2,0,0],body_pose=[[1,2,3]],expr=[0]*100)
        original=copy.deepcopy(packet)
        out=tracker.apply(packet,dict(mouthPucker=1.),0.)
        self.assertGreater(out['expr'][7],1)
        self.assertEqual(out['jaw_pose'],packet['jaw_pose'])
        self.assertEqual(out['body_pose'],packet['body_pose'])
        self.assertEqual(packet,original)
        before=tracker.previous.copy()
        with self.assertRaises(ValueError):tracker.apply(packet,dict(mouthPucker=float('nan')),10.)
        np.testing.assert_array_equal(tracker.previous,before)
        self.assertEqual(tracker.last_timestamp,0.)
        with self.assertRaises(ValueError):tracker.apply(packet,{},0.)

    def test_missing_scores_relax_and_output_bounded(self):
        tracker=LipExpressions(np.ones((100,1))*10,dict(names=['mouthPucker'],accepted=[True]))
        packet=dict(expr=[0]*100)
        first=tracker.apply(packet,dict(mouthPucker=1.),0.)
        second=tracker.apply(packet,{},100.)
        self.assertLess(np.linalg.norm(second['expr']),np.linalg.norm(first['expr']))
        self.assertLessEqual(np.linalg.norm(first['expr']),6.)


if __name__=='__main__':unittest.main()
