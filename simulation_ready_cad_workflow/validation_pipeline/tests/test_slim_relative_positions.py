"""Moving-reference coordinates retain actual body poses and real exits."""
import math
import unittest
from ..slim_harness import measurements as m

def pose(position,q=(1,0,0,0)):
 return {'position_abs_m':list(position),'rotation_abs_quat_wxyz':list(q)}

def samples():
 h=math.sqrt(.5)
 return [{'time_s':0.,'body_poses':{'tray':pose([0,0,0]),'bead':pose([.01,.02,.03])}},
         {'time_s':1.,'body_poses':{'tray':pose([.04,-.05,.06],[h,0,0,h]),'bead':pose([.02,-.04,.09])}},
         {'time_s':2.,'body_poses':{'tray':pose([.2,.1,-.1],[0,1,0,0]),'bead':pose([.21,.08,-.13])}},
         {'time_s':3.,'body_poses':{'tray':pose([1,2,3],[3*h,0,3*h,0]),'bead':pose([1.03,2.02,2.99])}}]
class Tests(unittest.TestCase):
 def test_translation_and_full_rotation_preserve_local_coordinates(self):
  for axis,value in zip('xyz',[.01,.02,.03]):
   result=m.evaluate(f'max(abs(relative_position("bead","tray","{axis}")-{value})) < 1e-14',{'time_s':[0.,1.,2.,3.]},pose_samples=samples())
   self.assertTrue(result['passed'],result)
 def test_actual_exit_is_detected_after_reference_moves(self):
  rows=samples();rows[2]['body_poses']['bead']['position_abs_m'][0]=.3
  result=m.evaluate('max(relative_position("bead","tray","x")) <= 0.015',{'time_s':[0.,1.,2.,3.]},pose_samples=rows)
  self.assertFalse(result['passed']);self.assertAlmostEqual(result['observed_operands'][0],.1)
 def test_window_retains_solver_times(self):
  result=m.evaluate('integral(window(relative_position("bead","tray","x"),1,3)) >= 0.0199999999',{'time_s':[0.,1.,2.,3.]},pose_samples=samples())
  self.assertTrue(result['passed']);self.assertAlmostEqual(result['observed_operands'][0],.02)
 def test_missing_or_unaligned_evidence_is_rejected(self):
  expr='first(relative_position("bead","tray","x")) >= 0'
  with self.assertRaisesRegex(ValueError,'recorded body poses'):m.evaluate(expr,{'time_s':[0.]})
  rows=samples();rows[1]['time_s']=1.1
  with self.assertRaisesRegex(ValueError,'match signal times'):m.evaluate(expr,{'time_s':[0.,1.,2.,3.]},pose_samples=rows)
  rows=samples();rows[2]['body_poses']['tray']['rotation_abs_quat_wxyz']=[0,0,0,0]
  with self.assertRaisesRegex(ValueError,'nonzero'):m.evaluate(expr,{'time_s':[0.,1.,2.,3.]},pose_samples=rows)
 def test_poses_are_read_once_for_all_assertions(self):
  class Rows:
   def __init__(self):self.reads=0;self.rows=samples()
   def __len__(self):return len(self.rows)
   def __iter__(self):self.reads+=1;return iter(self.rows)
  rows=Rows();test={'assertions':[{'id':a,'expression':f'max(abs(relative_position("bead","tray","{a}")-{v})) < 1e-14'} for a,v in zip('xyz',[.01,.02,.03])]}
  result=m.evaluate_assertions(test,{'time_s':[0.,1.,2.,3.]},pose_samples=rows)
  self.assertTrue(all(r['passed'] for r in result));self.assertEqual(rows.reads,1)
 def test_existing_signal_only_assertions_need_no_pose_evidence(self):
  result=m.evaluate_assertions({'assertions':[{'id':'old','expression':'span(signal("x")) == 2'}]},{'time_s':[0,1,2],'x':[0,1,2]})
  self.assertTrue(result[0]['passed'])

if __name__=='__main__':unittest.main()
