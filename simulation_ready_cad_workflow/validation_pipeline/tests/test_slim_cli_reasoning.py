"""Reasoning selection through public CLI initialization and resume; no physics."""
from contextlib import redirect_stdout
from importlib.util import find_spec
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ..mechanism.io import write_json_atomic
from ..slim_harness import cli
from . import test_budget_entrypoints as fixtures


class CliReasoningTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace=Path(self.temporary.name)

    def invoke(self,args,worker=None):
        worker=worker or fixtures.FakeWorker(author_status="ready")
        def supervised(workspace,project,**kwargs):
            return cli.resume_project(workspace,project,in_process=True,**kwargs)
        with patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.construction_supervisor.supervise_project",side_effect=supervised),patch.object(cli,"create_worker",return_value=worker) as factory,patch.object(cli,"SlimWorkflow") as workflow,redirect_stdout(io.StringIO()):
            workflow.return_value.run.return_value={"status":"NEEDS_REVISION","issues":[]}
            cli.main(["--workspace",str(self.workspace),*args])
        return factory,workflow

    def request(self,name):
        path=self.workspace/".runtime_runs/projects"/name/"workflow.json"
        return json.loads(path.read_text(encoding="utf-8"))["request"]

    def assert_workers(self,factory,reasoning):
        self.assertEqual(1,factory.call_count)
        self.assertEqual(reasoning,factory.call_args.kwargs["reasoning_effort"])
        self.assertEqual("gpt-6-astra",factory.call_args.kwargs["model"])

    def test_create_max_reaches_initial_author_and_workflow_then_resume_restores(self):
        factory,workflow=self.invoke(["--reasoning","max","--tier","light","create","--mode","free","--project","machine",
                                      "--brief","A slider must move."])
        self.assert_workers(factory,"max")
        self.assertEqual("max",workflow.call_args.kwargs["reasoning"])
        self.assertFalse(workflow.return_value.run.call_args.kwargs["one_revision"])
        self.assertEqual("max",self.request("machine")["reasoning_effort"])
        self.assertEqual("free",self.request("machine")["mode"])
        factory,workflow=self.invoke(["resume","machine"])
        factory.assert_not_called()
        self.assertEqual("max",workflow.call_args.kwargs["reasoning"])
        self.invoke(["--reasoning","high","resume","machine"])
        self.assertEqual("high",self.request("machine")["reasoning_effort"])
        _,workflow=self.invoke(["resume","machine"])
        self.assertEqual("high",workflow.call_args.kwargs["reasoning"])

    def test_submit_max_keeps_setting_without_calling_initial_worker(self):
        (self.workspace/"part.step").write_text("copy-only fixture; not geometric evidence")
        scene=self.workspace/"scene.json"
        write_json_atomic(scene,{"schema_version":"slim_scene_v1","units":"mm",
            "bodies":[{"id":"moving","source":"part.step","fixed":False}],
            "tests":[{"id":"motion","purpose":"Fixture","duration_s":.01,
                "assertions":[{"id":"a","purpose":"Fixture","expression":"last(signal('time_s')) >= 0.01"}]}]})
        worker=fixtures.FakeWorker(author_status="ready")
        factory,workflow=self.invoke(["--reasoning","max","submit",str(scene),"--mode","free","--project","submitted","--brief","A slider must move."],worker=worker)
        self.assert_workers(factory,"max")
        self.assertEqual([],worker.calls)
        self.assertEqual("max",workflow.call_args.kwargs["reasoning"])
        self.assertTrue(workflow.return_value.run.call_args.kwargs["one_revision"])
        self.assertEqual("max",self.request("submitted")["reasoning_effort"])

    def test_omitted_setting_uses_high_for_new_and_keeps_legacy_medium(self):
        factory,workflow=self.invoke(["create","--mode","free","--project","new","--brief","A slider must move."])
        self.assert_workers(factory,"high")
        self.assertEqual("high",workflow.call_args.kwargs["reasoning"])
        self.assertEqual("high",self.request("new")["reasoning_effort"])
        _,workflow=self.invoke(["resume","new"])
        self.assertEqual("high",workflow.call_args.kwargs["reasoning"])
        cli.create_project(self.workspace,"legacy","A slider must move.")
        factory,workflow=self.invoke(["resume","legacy"])
        self.assert_workers(factory,"medium")
        self.assertEqual("medium",workflow.call_args.kwargs["reasoning"])
        self.assertNotIn("reasoning_effort",self.request("legacy"))

    def test_existing_explicit_medium_is_not_upgraded_on_resume(self):
        self.invoke(["--reasoning","medium","create","--mode","free","--project","saved-medium","--brief","A slider must move."])
        _,workflow=self.invoke(["resume","saved-medium"])
        self.assertEqual("medium",workflow.call_args.kwargs["reasoning"])
        self.assertEqual("medium",self.request("saved-medium")["reasoning_effort"])

    def test_new_codex_and_cursor_projects_persist_independent_codex_reviewer_defaults(self):
        from ..cold_start.worker_factory import default_worker_model
        for provider in ('codex','cursor'):
            name='defaults-'+provider
            factory,_=self.invoke(['--worker-provider',provider,'create','--mode','free','--project',name,'--brief','A slider must move.'])
            request=self.request(name)
            self.assertEqual(default_worker_model(provider),request['model'])
            self.assertEqual('high' if provider=='codex' else 'medium',request['reasoning_effort'])
            self.assertEqual(request['reasoning_effort'],factory.call_args.kwargs['reasoning_effort'])
            self.assertEqual('codex',request['reviewer_provider'])
            self.assertEqual('gpt-6-astra',request['reviewer_model'])
            self.assertEqual('medium',request['reviewer_reasoning_effort'])
            self.invoke(['resume',name])
            self.assertEqual(request,self.request(name))

    def test_new_reviewer_overrides_and_legacy_missing_fields_are_preserved(self):
        self.invoke(['--worker-provider','cursor','create','--mode','free','--project','review-override',
            '--brief','A slider must move.','--reviewer-provider','cursor','--reviewer-model','explicit-review-model',
            '--reviewer-reasoning','xhigh','--reviewer-executable','explicit-review-cli'])
        request=self.request('review-override')
        self.assertEqual('cursor',request['reviewer_provider'])
        self.assertEqual('explicit-review-model',request['reviewer_model'])
        self.assertEqual('xhigh',request['reviewer_reasoning_effort'])
        self.assertEqual('explicit-review-cli',request['reviewer_executable'])
        self.invoke(['resume','review-override'])
        self.assertEqual(request,self.request('review-override'))
        cli.create_project(self.workspace,'old-reviewer','A slider must move.')
        self.invoke(['resume','old-reviewer'])
        self.assertFalse(any(name.startswith('reviewer_') for name in self.request('old-reviewer')))

    def test_new_task_defaults_high_and_resume_preserves_saved_tier(self):
        for command in ('create','submit'):
            extra=[] if command=='create' else ['scene.json']
            with self.subTest(command=command), patch.object(cli,'start',return_value={'status':'PASS'}) as launch,redirect_stdout(io.StringIO()):
                cli.main([command,*extra,'--mode','free','--project','default','--brief','Task'])
                self.assertEqual('high',launch.call_args.kwargs['tier'])
        with patch.object(cli,'start',return_value={'status':'PASS'}) as launch,redirect_stdout(io.StringIO()):
            cli.main(['--tier','light','create','--mode','free','--project','explicit','--brief','Task'])
            self.assertEqual('light',launch.call_args.kwargs['tier'])
        with patch.object(cli,'resume_project',return_value={'status':'PASS'}) as resume,redirect_stdout(io.StringIO()):
            cli.main(['resume','existing'])
            self.assertIsNone(resume.call_args.kwargs['tier'])
        with patch.object(cli,'start',return_value={'status':'PASS'}) as launch,redirect_stdout(io.StringIO()):
            cli.main(['create','--mode','free','--construction','--budget-hours','2','--project','special','--brief','Task'])
            self.assertEqual('high',launch.call_args.kwargs['tier'])
            self.assertEqual(2,launch.call_args.kwargs['budget_config'].hours)

    def test_resume_incomplete_initial_author_restores_custom_model(self):
        factory,_=self.invoke(['--model','custom-model','create','--mode','free','--project','custom','--brief','A slider must move.'],
                             worker=fixtures.FakeWorker(author_status='needs_revision'))
        self.assertEqual('custom-model',factory.call_args.kwargs['model'])
        factory,workflow=self.invoke(['resume','custom'])
        self.assertEqual('custom-model',factory.call_args.kwargs['model'])
        self.assertEqual('custom-model',workflow.call_args.kwargs['model'])

    def test_resume_incomplete_initial_author_restores_saved_max(self):
        factory,workflow=self.invoke(["--reasoning","max","create","--mode","free","--project","recover","--brief","A slider must move."],
                                     worker=fixtures.FakeWorker(author_status="needs_revision"))
        self.assert_workers(factory,"max")
        workflow.assert_not_called()
        factory,workflow=self.invoke(["resume","recover"])
        self.assert_workers(factory,"max")
        self.assertEqual("max",workflow.call_args.kwargs["reasoning"])
        self.assertEqual("max",self.request("recover")["reasoning_effort"])

    @unittest.skipUnless(find_spec('simulation_ready_cad_workflow.validation_pipeline.benchmark_sandbox.formal_compute'),
                         'The no-visuals ablation needs the formal sandbox')
    def test_author_visual_policy_persisted_on_new_project_and_resume(self):
        self.invoke(['--author-visual-policy','worker_no_visuals_v1','create','--mode','free',
                     '--project','text-only','--brief','A slider must move.'])
        self.assertEqual('worker_no_visuals_v1',self.request('text-only')['author_visual_policy'])
        self.invoke(['resume','text-only'])
        self.assertEqual('worker_no_visuals_v1',self.request('text-only')['author_visual_policy'])
        with self.assertRaises(SystemExit):
            self.invoke(['--author-visual-policy','full','resume','text-only'])

    def test_unsupported_visual_policy_client_rejected_before_project_creation(self):
        with self.assertRaisesRegex(ValueError,'original Codex'):
            self.invoke(['--worker-provider','cursor','--author-visual-policy','worker_no_visuals_v1',
                         'create','--mode','free','--project','unsupported-visual','--brief','Task'])
        self.assertFalse((self.workspace/'.runtime_runs/projects/unsupported-visual').exists())


if __name__=="__main__":unittest.main()
