"""Pure template orchestration and native-binding interface checks; no solver steps."""
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from benchmark_templates import runtime
from benchmark_templates.native import Signal
from ..mechanism.io import write_json_atomic as write
from ..slim_harness import template_tasks, template_workflow as workflow

PREFIX = 'simulation_ready_cad_workflow.validation_pipeline.'
IMPLEMENTATION_BINDING = workflow.implementation_binding


class BindingPrecheckTests(unittest.TestCase):
    def setUp(self):
        directory = Path(__file__).resolve().parents[3]/'.runtime_runs/preflight_20260916'
        directory.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=directory)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prepared = {'bodies': [{'id': 'shaft'}]}
        write(self.root / 'prepared.json', self.prepared)
        self.created = []
        self.signals = {'shaft.speed': Signal(lambda: 0.0, 'source_scalar')}
        self.actuators = {'drive': object()}
        trial = SimpleNamespace(id='load', required_signals=(), checks=(SimpleNamespace(signals=lambda: {'shaft.speed'}),),
                                actions=(runtime.Action('drive', 0, 1, 'effort', 1, 1),))
        self.template = SimpleNamespace(required_signals=(), trials=(trial,))
        self.request = {'prepared': str(self.root/'prepared.json'), 'output': str(self.root/'check'), 'step_s': .001}

        def build(*args):
            # There is deliberately no stepping API on this system.
            system = SimpleNamespace(GetChTime=lambda: 0.0)
            bodies = {'shaft': object()}
            hooks = dict(before_step=None, after_step=None, commit_inputs=None)
            self.created.append({'system': system, 'bodies': bodies, **hooks})
            return SimpleNamespace(system=system, objects={'prepared': self.prepared, 'bodies': bodies},
                                   signals=self.signals, actuators=self.actuators, **hooks)
        self.binding = SimpleNamespace(build_model=build)

    def check(self):
        template_tasks.precheck_binding(self.template, self.binding, {}, self.request, runtime, self.created)

    def test_valid_original_model_is_checked_without_stepping(self):
        self.check()
        value = workflow.read(self.root/'check/precheck.json')
        self.assertEqual(value['status'], 'ready')
        self.assertFalse(value['stepped'])
        self.assertTrue(workflow.read(self.root/'check/load/native-model-binding.json')['source_model_retained'])

    def test_missing_required_observable_fails_before_execution(self):
        self.signals.clear()
        with self.assertRaisesRegex(ValueError, 'Missing required observables'):
            self.check()

    def test_missing_actuator_fails_before_execution(self):
        self.actuators.clear()
        with self.assertRaisesRegex(ValueError, 'Missing actuators'):
            self.check()

    def test_broken_initial_signal_fails_before_execution(self):
        self.signals['shaft.speed'] = Signal(lambda: float('nan'), 'source_scalar')
        with self.assertRaisesRegex((ValueError, RuntimeError), 'Nonfinite native signal|Non-finite initial observable'):
            self.check()


class TemplateWorkflowPrecheckTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[3]
        directory = root/'.runtime_runs/preflight_20260916'
        directory.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=directory)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        project = self.root/'project'
        design = project/'runs/r01/design'
        design.mkdir(parents=True)
        (design/'part.step').write_text('fixture; never opened by CAD')
        write(design/'scene.json', {'schema_version':'slim_scene_v1', 'units':'m', 'bodies':[{'id':'part','source':'part.step'}]})
        (project/'task.md').write_text('fixture task')
        write(project/'criteria.json', {'fixture':True})
        write(project/'mode.json', {'mode':'free'})
        (design/'task_template.py').write_text('# fixture template')
        self.design = design
        self.source_before = (design/'scene.json').read_bytes()
        self.plan = {'trials':[{'id':name,'duration_s':1.0,'checks':[{'id':'check'}]} for name in ('first','second')]}
        self.flow = SimpleNamespace(project=project, workspace=root, state={'revision':1}, chrono_python='fixture-python',
                                    tier=None, construction=None, building=False, physics_timeout_seconds=300)
        self.flow.criteria_sha256 = workflow.file_digest(project/'criteria.json',drop_cache=True)
        self.flow.budget = Mock()
        self.flow.budget.limit.side_effect = lambda x:x
        self.flow.budget.active.side_effect = lambda x:nullcontext()
        self.flow.publish = Mock()
        self.flow._preview = Mock()
        self.flow.reviewer = SimpleNamespace(run_turn=self.review)
        self.flow.turn = lambda kind,call:call(self.root/'turn')
        self.review_calls = 0
        self.precheck_failures = 0
        self.prepare_exit = None
        self.operations = []
        self.runtime_status = 'binding_or_execution_error'
        self.limits = []
        class Capture:
            def __init__(self,*args,**kwargs): Path(args[1]).mkdir(parents=True,exist_ok=True)
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def command(self,*args,**kwargs): return nullcontext({})
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for target,value in [
            ('slim_harness.template_workflow.implementation_binding',lambda:{}),
            ('slim_harness.template_workflow.run_bounded_subprocess',self.command),
            ('slim_harness.physical_parts.resolve_physical_part_inventory',lambda *a,**k:{}),
            ('slim_harness.physical_parts.check_physical_part_inventory',lambda *a,**k:[]),
            ('slim_harness.static_source_metadata.grouped_manifest_findings',lambda *a,**k:[]),
            ('slim_harness.print_delivery.load_registered_static_evidence',lambda *a,**k:({},[])),
            ('slim_harness.step_delivery.refresh_registered_step_delivery',lambda *a,**k:None),
            ('trajectory.capture.HarnessProgramCapture',Capture),
            ('trajectory.capture.publish_attempt',lambda *a,**k:None),
        ]: self.stack.enter_context(patch(PREFIX+target,value))

    def review(self,binding_dir,prompt,*args,**kwargs):
        self.review_calls += 1
        prepared = binding_dir.parent/'prepared/prepared.json'
        self.assertTrue(prepared.is_file(), 'Reviewer must receive an already prepared model')
        self.assertIn(str(prepared),prompt)
        (binding_dir/'binding.py').write_text('# independently filled fixture')
        return {'status':'ready'}

    def command(self,argv,**kwargs):
        operation = argv[-2]
        request = workflow.read(argv[-1])
        self.operations.append((operation,request.get('trial')))
        if self.flow.construction is not None:
            sessions = workflow.read(self.flow.project/'construction_budget.json')['sessions'].values()
            self.limits.append((operation,kwargs['timeout_seconds'],sorted(session['kind'] for session in sessions)))
        output = Path(request['output'])
        if operation == 'prepare' and self.prepare_exit is not None: return self.prepare_exit
        if operation == 'plan': write(output,self.plan)
        elif operation == 'prepare':
            write(output/'prepared.json',{'bodies':[{'id':'part'}]})
            write(output/'geometry/geometry.json',{'bodies':[{'id':'part'}]})
        elif operation == 'audit': write(output,{'passed':True})
        elif operation == 'precheck':
            if self.precheck_failures:
                self.precheck_failures -= 1
                raise RuntimeError('Missing actuator: drive')
            write(output/'precheck.json',{'status':'ready','stepped':False})
        elif operation == 'run':
            write(output/'summary.json',{'trials':[{'id':request['trial'],'status':self.runtime_status,'checks':[],
                                                  'functional_pass':None,'error':'fixture technical error'}]})
        else: raise AssertionError(operation)
        return SimpleNamespace(returncode=0,stdout=b'',stderr=b'')

    def test_reviewer_repairs_precheck_on_same_prepared_source_and_trials_keep_first_opportunity(self):
        self.precheck_failures = 1
        result = workflow.run_revision(self.flow)
        self.assertEqual(self.review_calls,2)
        self.assertEqual(sum(op == 'prepare' for op,_ in self.operations),1)
        self.assertEqual([trial for op,trial in self.operations if op == 'run'],['first','second'])
        self.assertEqual(result['status'],'EVIDENCE_REQUIRED')
        self.assertTrue(result['template_binding_error'])
        self.assertEqual((self.design/'scene.json').read_bytes(),self.source_before)

    def test_construction_trials_run_uncharged_with_the_extended_limit(self):
        from ..slim_harness.construction_budget import BudgetConfig, initialize_budget
        self.flow.construction = initialize_budget(self.flow.project, BudgetConfig(hours=1, simulation_timeout_seconds=600))
        self.flow.building = True
        workflow.run_revision(self.flow)
        trials = [(timeout,kinds) for operation,timeout,kinds in self.limits if operation == 'run']
        self.assertEqual([(1800.,['simulation'])]*2,trials)
        self.assertTrue(all('simulation' not in kinds for operation,_,kinds in self.limits if operation != 'run'))
        rows = workflow.read(next(self.flow.project.glob('runs/r01/template_runs/a*/template-result.json')))['tests']
        self.assertEqual([(600.,[600.,1200.,1800.],None)]*2,
                         [(row['configured_timeout_seconds'],row['timeout_steps_seconds'],row['extended_to_seconds']) for row in rows])

    def test_exhausted_binding_precheck_does_not_run_physics_or_request_mechanical_revision(self):
        self.precheck_failures = 2
        result = workflow.run_revision(self.flow)
        self.assertEqual(self.review_calls,2)
        self.assertFalse(any(op == 'run' for op,_ in self.operations))
        self.assertEqual(result['status'],'EVIDENCE_REQUIRED')
        self.assertTrue(result['template_binding_error'])

    def test_assessment_error_remains_technical(self):
        self.runtime_status = 'assessment_error'
        result = workflow.run_revision(self.flow)
        self.assertEqual(result['status'],'EVIDENCE_REQUIRED')
        self.assertTrue(result['template_binding_error'])
        self.assertEqual([row['execution_status'] for row in result['tests']],['assessment_error']*2)

    def failed_prepare(self, message):
        self.prepare_exit = SimpleNamespace(returncode=template_tasks.DELIVERY_ERROR_EXIT, stderr=b'Traceback (fixture)',
                                            stdout=json.dumps({'delivery_error': message}).encode())
        return workflow.run_revision(self.flow)

    def test_author_delivery_error_in_prepare_returns_to_the_author(self):
        message = 'Source support: Declared bore surface is absent from the original source'
        result = self.failed_prepare(message)
        self.assertEqual((result['status'], result['issues']), ('NEEDS_REVISION', [message]))
        self.assertEqual(self.review_calls, 0)

    def test_capability_gap_from_prepare_remains_a_harness_error(self):
        result = self.failed_prepare('CAPABILITY_GAP: no native model for this joint')
        self.assertEqual(result['status'], 'HARNESS_ERROR')

    def test_data_catalogue_cannot_replace_loaded_implementation_provenance(self):
        data = self.root / 'data'
        catalogue = data / 'benchmark_templates'
        catalogue.mkdir(parents=True)
        (catalogue / 'native.py').write_text('# stale data catalogue; never imported\n')
        self.flow.workspace = data
        with patch(PREFIX+'slim_harness.template_workflow.implementation_binding', IMPLEMENTATION_BINDING):
            result = workflow.run_revision(self.flow)
        self.assertEqual(result['status'], 'EVIDENCE_REQUIRED')
        self.assertEqual(result['execution_implementation'], IMPLEMENTATION_BINDING())
        self.assertFalse(any(Path(name).is_relative_to(data) for name in result['execution_implementation']))
        self.assertIn(str(Path(runtime.__file__).resolve()), result['execution_implementation'])


class DeliveryErrorExitTests(unittest.TestCase):
    def test_prepare_reports_an_author_delivery_error_across_the_process_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root/'scene.json', {'schema_version': 'slim_scene_v1', 'units': 'm', 'bodies': []})
            write(root/'request.json', {'scene': str(root/'scene.json'), 'output': str(root/'out')})
            completed = subprocess.run([sys.executable, '-m', PREFIX + 'slim_harness.template_tasks', 'prepare',
                                        str(root/'request.json')], cwd=Path(__file__).resolve().parents[3],
                                       capture_output=True, text=True, timeout=300)
        self.assertEqual(completed.returncode, template_tasks.DELIVERY_ERROR_EXIT, completed.stderr)
        self.assertIn('physical body', json.loads(completed.stdout.splitlines()[-1])['delivery_error'])


class ConstructorParameterMatchTests(unittest.TestCase):
    def test_accepts_value_reason_objects_and_ignores_unused_names(self):
        protocol = {"drive_s": 2.0, "mass_kg": 1.0}
        requested = {
            "drive_s": {"value": 2.0, "reason": "shorter trial"},
            "mass_kg": 1.0,
            "unused_press_n": 40,
        }
        self.assertTrue(template_tasks.constructor_parameters_match(protocol, requested))
        requested["drive_s"] = {"value": 3.0, "reason": "changed"}
        self.assertFalse(template_tasks.constructor_parameters_match(protocol, requested))


if __name__ == '__main__': unittest.main()
