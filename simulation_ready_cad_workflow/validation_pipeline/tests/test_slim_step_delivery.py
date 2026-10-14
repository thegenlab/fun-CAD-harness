"""Real small STEP rebuilds; no printing, model calls, or mechanical verdict."""
import json
from pathlib import Path
import runpy
import shutil
import tempfile
import unittest
from unittest.mock import patch

from ..slim_harness.asset import build_file
from ..slim_harness.cli import create_project
from ..slim_harness.static_delivery import digest,write
from ..slim_harness.step_delivery import ensure_step_delivery,load_registered_step_evidence,refresh_registered_step_delivery
from ..slim_harness.print_delivery import load_registered_static_evidence,is_static_delivery_criterion

SOURCE='''from build123d import Box,Compound,Pos
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.asset import Asset,Part
def shapes():return {'base':Box(10,10,2),'cap':Pos(0,0,4)*Box(4,4,2)}
def gen_step():return Compound(children=list(shapes().values()))
def build():return Asset(parts={name:Part(shape,body='one_body',role='fixed') for name,shape in shapes().items()},dependencies=['assembly.step','OPERATING.md'])
'''


class StepDeliveryTests(unittest.TestCase):
    def setUp(self):
        import build123d as b
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.workspace=Path(self.temp.name)
        self.project=create_project(self.workspace,'step_fixture','Deliver a rebuildable STEP assembly and demonstrate motion.')
        self.design=self.project/'runs/r01/design'
        self.source=self.design/'model.py';self.source.write_text(SOURCE,encoding='utf-8')
        (self.design/'OPERATING.md').write_text('Fixture operating instructions.',encoding='utf-8')
        b.export_step(runpy.run_path(str(self.source))['gen_step'](),self.design/'assembly.step')
        build_file(self.source,self.design)
        criterion={'id':'delivery','kind':'function','statement':'Deliver source and demonstrate motion.',
                   'source_quote':'Deliver a rebuildable STEP assembly and demonstrate motion.','verification':'Rebuild and separately test motion.'}
        write(self.project/'criteria.json',{'task_text':(self.project/'task.md').read_text(encoding='utf-8'),'criteria':[criterion]})
        write(self.project/'criteria.lock.json',{'criteria_sha256':digest(self.project/'criteria.json'),'task_sha256':digest(self.project/'task.md')})
        self.attempt=self.project/'runs/r01/attempt';self.attempt.mkdir()
        self.requests=[{'criterion_id':'delivery','action':'local_measurement','target':'step_delivery:supplemental'}]

    def test_step_only_rebuild_and_grouped_parts_do_not_certify_motion(self):
        result=ensure_step_delivery(self.project,self.design/'scene.json',self.attempt,self.requests)
        self.assertEqual(result['status'],'ready',result)
        evidence,errors=load_registered_static_evidence(self.project,self.design/'scene.json')
        self.assertFalse(errors);self.assertEqual(len(evidence),1)
        criterion=json.loads(evidence[0].criterion_json)
        self.assertFalse(is_static_delivery_criterion(criterion,evidence))
        inspection=json.loads(Path(result['evidence_paths'][1]).read_text(encoding='utf-8'))
        self.assertFalse(inspection['print_assets_required'])
        self.assertFalse(inspection['documentation_semantics_verified'])
        self.assertEqual([row['solid_count'] for row in inspection['step_files']],[2,2])
        again=ensure_step_delivery(self.project,self.design/'scene.json',self.attempt,self.requests)
        self.assertEqual(again['receipt_ids'],result['receipt_ids'])

    def test_changed_source_invalidates_registered_receipt(self):
        result=ensure_step_delivery(self.project,self.design/'scene.json',self.attempt,self.requests)
        self.assertEqual(result['status'],'ready')
        self.source.write_text(SOURCE+'\n# changed\n',encoding='utf-8')
        evidence,errors=load_registered_step_evidence(self.project,self.design/'scene.json')
        self.assertFalse(evidence);self.assertTrue(errors)

    def test_declared_nested_assembly_output_is_created(self):
        (self.design/'exports').mkdir()
        (self.design/'assembly.step').rename(self.design/'exports/assembly.step')
        self.source.write_text(SOURCE.replace('assembly.step','exports/assembly.step'),encoding='utf-8')
        build_file(self.source,self.design)
        result=ensure_step_delivery(self.project,self.design/'scene.json',self.attempt,self.requests)
        self.assertEqual(result['status'],'ready',result)

    def test_different_delivered_geometry_cannot_get_receipt(self):
        import build123d as b
        b.export_step(b.Box(11,10,2),self.design/'assembly.step')
        result=ensure_step_delivery(self.project,self.design/'scene.json',self.attempt,self.requests)
        self.assertEqual(result['status'],'needs_revision')
        self.assertFalse((self.project/'step-delivery-registration.json').exists())

    def test_new_revision_rebuilds_and_registers_without_changing_old_receipt(self):
        result=ensure_step_delivery(self.project,self.design/'scene.json',self.attempt,self.requests)
        self.assertEqual('ready',result['status'])
        registration=self.project/'step-delivery-registration.json'
        old=json.loads(registration.read_text());old_receipt=Path(old['receipt_path']).read_bytes()
        design=self.project/'runs/r02/design';shutil.copytree(self.design,design)
        attempt=design.parent/'attempt'
        result=refresh_registered_step_delivery(self.project,design/'scene.json',attempt)
        self.assertEqual('ready',result['status'],result)
        evidence,errors=load_registered_step_evidence(self.project,design/'scene.json')
        self.assertFalse(errors);self.assertEqual(1,len(evidence))
        self.assertEqual(str(design/'scene.json'),evidence[0].scene_path)
        self.assertEqual(old_receipt,Path(old['receipt_path']).read_bytes())
        self.assertEqual(old,json.loads((attempt/'step-registration.before.json').read_text()))
        with patch('simulation_ready_cad_workflow.validation_pipeline.slim_harness.step_delivery.execute',side_effect=AssertionError('Valid receipt must not rebuild')):
            self.assertIsNone(refresh_registered_step_delivery(self.project,design/'scene.json',attempt))

    def test_changed_criterion_is_not_silently_remapped_by_id(self):
        ensure_step_delivery(self.project,self.design/'scene.json',self.attempt,self.requests)
        registration=self.project/'step-delivery-registration.json';before=registration.read_bytes()
        criteria=json.loads((self.project/'criteria.json').read_text())
        criteria['criteria'][0]['statement']='Different requirement with the same ID'
        write(self.project/'criteria.json',criteria)
        result=refresh_registered_step_delivery(self.project,self.design/'scene.json',self.attempt/'refresh')
        self.assertEqual('needs_revision',result['status'])
        self.assertIn('criteria changed',result['issues'][0])
        self.assertEqual(before,registration.read_bytes())

    def test_no_existing_registration_does_not_invent_delivery_requirements(self):
        with patch('simulation_ready_cad_workflow.validation_pipeline.slim_harness.step_delivery.execute',side_effect=AssertionError('No unsolicited rebuild')):
            self.assertIsNone(refresh_registered_step_delivery(self.project,self.design/'scene.json',self.attempt))

    def test_failed_refresh_keeps_prior_registration_and_does_not_accept_changed_geometry(self):
        import build123d as b
        ensure_step_delivery(self.project,self.design/'scene.json',self.attempt,self.requests)
        registration=self.project/'step-delivery-registration.json';before=registration.read_bytes()
        b.export_step(b.Box(11,10,2),self.design/'assembly.step')
        result=refresh_registered_step_delivery(self.project,self.design/'scene.json',self.attempt/'refresh')
        self.assertEqual('needs_revision',result['status'])
        self.assertEqual(before,registration.read_bytes())
        evidence,errors=load_registered_step_evidence(self.project,self.design/'scene.json')
        self.assertFalse(evidence);self.assertTrue(errors)


if __name__=='__main__':unittest.main()
