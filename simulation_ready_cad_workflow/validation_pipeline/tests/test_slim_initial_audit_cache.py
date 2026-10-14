"""Offline exact-initial-state reuse tests; never run CAD or audit_motion."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest

from simulation_ready_cad_workflow.validation_pipeline.slim_harness.workflow import reuse_initial_geometry_audit as reuse, AUDIT_IMPLEMENTATION_SHA256


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


class InitialAuditReuse(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.attempt=Path(temporary.name)/'a002';self.audit=self.attempt/'initial-geometry-audit.json'
        self.mesh=self.attempt/'geometry/source-component.obj';self.mesh.parent.mkdir(parents=True)
        self.mesh.write_text('synthetic immutable component mesh fixture',encoding='utf-8')
        self.geometry={'bodies':[{'id':'part','position_m':[1.,2.,3.],'rotation_wxyz':[1.,0.,0.,0.],
            'source_components':[{'mesh_file':str(self.mesh),'sha256':digest(self.mesh)}]}]}
        write(self.attempt/'geometry/geometry.json',self.geometry)
        self.report={'schema_version':'slim_geometry_audit_v1','audit_implementation_sha256':AUDIT_IMPLEMENTATION_SHA256,
            'passed':True,'violations':[],'checked_solver_samples':1,'available_solver_samples':1,
            'elapsed_seconds':864.2766,'scope':'original geometry at every solver state; no claim between solver steps'}
        write(self.audit,self.report)

    def test_exact_original_pose_reuses_without_mutating_prior_report(self):
        original=self.audit.read_bytes();result=reuse(copy.deepcopy(self.geometry),self.attempt)
        self.assertTrue(result['passed']);self.assertEqual(str(self.audit),result['reused_from'])
        self.assertEqual(864.2766,result['original_execution_elapsed_seconds'])
        self.assertGreaterEqual(result['elapsed_seconds'],0)
        self.assertEqual(original,self.audit.read_bytes())

    def test_changed_audit_implementation_does_not_reuse(self):
        self.report['audit_implementation_sha256']='old-audit';write(self.audit,self.report)
        self.assertIsNone(reuse(self.geometry,self.attempt))

    def test_moved_or_rotated_initial_body_does_not_reuse(self):
        for field,value in [('position_m',[1.,2.,3.1]),('rotation_wxyz',[0.,0.,0.,1.])]:
            moved=copy.deepcopy(self.geometry);moved['bodies'][0][field]=value
            with self.subTest(field=field):self.assertIsNone(reuse(moved,self.attempt))

    def test_component_content_change_or_missing_file_does_not_reuse(self):
        self.mesh.write_text('changed component mesh',encoding='utf-8')
        self.assertIsNone(reuse(self.geometry,self.attempt))
        self.mesh.unlink()
        self.assertIsNone(reuse(self.geometry,self.attempt))

    def test_nonpass_multisample_or_excluded_scope_does_not_reuse(self):
        changes=[{'passed':False},{'violations':[{'kind':'overlap'}]},
                 {'checked_solver_samples':2},{'available_solver_samples':2},
                 {'schema_version':'slim_geometry_audit_v2'}, {'validation_scope':{}}, {'validation_scope':None}]
        for change in changes:
            write(self.audit,{**self.report,**change})
            with self.subTest(change=change):self.assertIsNone(reuse(self.geometry,self.attempt))

    def test_no_cache_or_missing_report_does_not_reuse(self):
        self.assertIsNone(reuse(self.geometry,None))
        self.audit.unlink();self.assertIsNone(reuse(self.geometry,self.attempt))

    def test_repeated_reuse_preserves_original_execution_time(self):
        write(self.audit,{**self.report,'elapsed_seconds':.02,'original_execution_elapsed_seconds':864.2766,
                          'reused_from':'previous-attempt/initial-geometry-audit.json'})
        result=reuse(self.geometry,self.attempt)
        self.assertEqual(864.2766,result['original_execution_elapsed_seconds'])


if __name__=='__main__':unittest.main()
