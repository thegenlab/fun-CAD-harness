"""Direct body motor/brake boundaries without an internal ideal joint."""
import os
from pathlib import Path
import subprocess
import unittest

CHRONO=Path(os.environ.get('SIMREADY_TEST_CHRONO_PYTHON',
    str(Path.home() / ".conda/envs/chrono_py312_projectchrono/python.exe")))


@unittest.skipUnless(CHRONO.is_file(),'Native Chrono interpreter not configured')
class BodyBoundaryLoadTests(unittest.TestCase):
    def test_jointless_motor_and_passive_brake_account_for_actual_work(self):
        import tempfile,trimesh
        with tempfile.TemporaryDirectory() as d:
            mesh=Path(d)/'body.obj';trimesh.creation.box(extents=[.1,.1,.1]).export(mesh)
            script=r'''
import json,sys
from pathlib import Path
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.runtime import simulate
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.trace_storage import open_trace
p=Path(sys.argv[1]);inertia=.1
body={'id':'rotor','fixed':False,'position_m':[0,0,0],'rotation_wxyz':[1,0,0,0],
      'mass_kg':1.,'inertia_kg_m2':[[inertia,0,0],[0,inertia,0],[0,0,inertia]],'mesh_file':str(p/'body.obj'),'source_solid_ids':['solid_1']}
test={'id':'motor_brake','purpose':'Actual body-boundary work','duration_s':.1,'step_s':.001,'inputs':[],
      'loads':[{'id':'motor','body':'rotor','torque_nm':[0,0,1],'end_s':.05},
               {'id':'brake','body':'rotor','rotational_damping_nm_s_rad':.1,'axis_abs':[0,0,1]}],
      'assertions':[{'id':'rotation','purpose':'Finite motor rotates the free body','expression':"last(signal('rotor.angular_speed.z_rad_s')) > 0"}]}
simulate({'bodies':[body],'joints':[],'gravity_m_s2':[0,0,0]},test,p/'trace.json')
t=open_trace(p/'trace.json');s=t['signals'];omega=s['rotor.angular_speed.z_rad_s'][-1]
work=s['load.motor.work_j'][-1]+s['load.brake.work_j'][-1]
assert t['physical_pass'] and s['load.motor.work_j'][-1]>0 and s['load.brake.work_j'][-1]<0
assert abs(s['load.brake.work_j'][-1]+s['load.brake.dissipated_work_j'][-1])<1e-12
assert abs(work-.5*inertia*omega**2)<1e-8,(work,omega)
print('Jointless body motor/brake physics and work accounting passed')
'''
            r=subprocess.run([str(CHRONO),'-c',script,d],cwd=Path(__file__).resolve().parents[3],capture_output=True,text=True,timeout=60)
            self.assertEqual(r.returncode,0,r.stderr+r.stdout)
