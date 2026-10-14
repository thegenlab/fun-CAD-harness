"""Control inputs use half-open windows; force samples describe completed steps."""
from pathlib import Path
import subprocess
import os
import tempfile
import unittest

from ..slim_harness.runtime import _active_at

CHRONO=Path(os.environ.get('SIMREADY_TEST_CHRONO_PYTHON',str(Path.home() / ".conda/envs/chrono_py312_projectchrono/python.exe")))

class InputWindowTests(unittest.TestCase):
    def test_declared_end_excludes_the_step_starting_at_that_time(self):
        interval={'start_s':13.,'end_s':28.}
        self.assertTrue(_active_at(interval,13.,40.5))
        self.assertTrue(_active_at(interval,27.999,40.5))
        self.assertFalse(_active_at(interval,28.,40.5))
        self.assertFalse(_active_at(interval,28.001,40.5))

    def test_default_end_and_zero_duration(self):
        self.assertTrue(_active_at({},.009,.01))
        self.assertFalse(_active_at({},.01,.01))
        self.assertFalse(_active_at({'start_s':.005,'end_s':.005},.005,.01))

    @unittest.skipUnless(CHRONO.is_file(),'Verified Chrono runtime unavailable')
    def test_drive_and_body_load_stop_before_next_recorded_step(self):
        import trimesh
        workspace=Path(__file__).resolve().parents[3]
        root=workspace/'.runtime_runs/input_window_regression';root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            p=Path(temporary);trimesh.creation.box(extents=[.1,.1,.1]).export(p/'body.obj')
            script=r"""
import sys,numpy as np
from pathlib import Path
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.runtime import simulate
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.trace_storage import open_trace
p=Path(sys.argv[1])
def body(name,position,fixed):
 return {'id':name,'fixed':fixed,'position_m':position,'rotation_wxyz':[1,0,0,0],'mass_kg':1.,'inertia_kg_m2':np.diag([.1,.1,.1]).tolist(),'mesh_file':str(p/'body.obj'),'source_solid_ids':['solid_1']}
a=body('rotor',[0,0,0],False);b=body('ground',[0,-.2,0],True)
j={'id':'j','type':'revolute','body_a':'rotor','body_b':'ground','anchor':[0,0,0],'axis':[0,0,1],
   'region_a':{'points':[[0,0,0]]*2,'radius':.4},'region_b':{'points':[[0,.2,0]]*2,'radius':.4}}
test={'id':'input_window','duration_s':.006,'step_s':.001,'inputs':[{'joint':'j','mode':'effort','value':.002,'max_effort':.002,'start_s':0,'end_s':.003}],
      'loads':[{'id':'body_motor','body':'rotor','torque_nm':[0,0,.001],'start_s':0,'end_s':.003}],'assertions':[]}
simulate({'bodies':[a,b],'joints':[j],'gravity_m_s2':[0,0,0]},test,p/'trace.json')
t=open_trace(p/'trace.json');s=t['signals'];assert len(t['samples'])==7 and t['physical_pass']
assert s['input.j.effort'][3]==.002
for i in (4,5,6):
 assert s['input.j.effort'][i]==0.,(i,s['input.j.effort'][i])
 evidence=t['samples'][i]['force_evidence'];assert evidence['step_start_s']>=.003
 assert not [row for row in evidence['external_loads'] if row['kind']!='gravity'],evidence['external_loads']
 assert s['input.j.work_j'][i]==s['input.j.work_j'][3]
 assert s['load.body_motor.work_j'][i]==s['load.body_motor.work_j'][3]
 assert abs(s['rotor.angular_speed.z_rad_s'][i]-s['rotor.angular_speed.z_rad_s'][3])<1e-12
print('Exclusive input end and cleared native loads verified')
"""
            result=subprocess.run([str(CHRONO),'-X','utf8','-c',script,str(p)],cwd=workspace,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)

if __name__=='__main__':unittest.main()
