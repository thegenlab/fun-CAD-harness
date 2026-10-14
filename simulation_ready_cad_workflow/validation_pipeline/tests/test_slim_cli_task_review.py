"""Public CLI binds the original task without initial model review or physics."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ..slim_harness import cli


class CapturingAuthor:
    session_id="fixture"
    def __init__(self,status="ready"):
        self.prompts=[]
        self.status=status
    def run_turn(self,workspace,prompt,directory,timeout_seconds,**kwargs):
        self.prompts.append(prompt)
        return {"status":self.status,"summary":"Author fixture","issues":[],"evidence_paths":[]}


class InitialTaskBindingCliTests(unittest.TestCase):
    original="Build a useful slider mechanism.\nTravel at least 10 mm under a 2 N load."

    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.workspace=Path(self.temporary.name)

    def create(self,name,*,adjustable=False,author_status="ready"):
        author=CapturingAuthor(author_status)
        arguments=["--workspace",str(self.workspace),"--reasoning","max","--tier","light"]
        arguments += ["create","--mode","free","--project",name,"--brief",self.original]
        with patch.object(cli,"create_worker",return_value=author) as factory,patch.object(cli,"SlimWorkflow") as workflow,redirect_stdout(io.StringIO()):
            workflow.return_value.run.return_value={"status":"NEEDS_REVISION","issues":[]}
            if adjustable:
                # Retain coverage of saved legacy projects through the Python entrypoint.
                cli.start(self.workspace,name,self.original,reasoning="max",tier="light",task_adjustable=True)
            else:
                cli.main(arguments)
        self.assertEqual(1,factory.call_count)
        self.assertEqual(1,len(author.prompts))
        return self.workspace/".runtime_runs/projects"/name,author,workflow

    def test_create_binds_complete_original_task_without_initial_model_review(self):
        for adjustable in (False,True):
            with self.subTest(adjustable=adjustable):
                project,author,workflow=self.create(f"binding_{adjustable}",adjustable=adjustable)
                self.assertEqual(self.original,(project/"task.md").read_text())
                criteria=json.loads((project/"criteria.json").read_text())
                self.assertEqual(self.original,criteria["task_text"])
                self.assertEqual("program_task_binding",criteria["source"])
                self.assertEqual(1,len(criteria["criteria"]))
                criterion=criteria["criteria"][0]
                self.assertEqual(self.original,criterion["statement"])
                self.assertEqual(self.original,criterion["source_quote"])
                state=json.loads((project/"workflow.json").read_text())
                self.assertEqual(self.original,state["request"]["brief"])
                self.assertEqual(["criteria","author"],[event["kind"] for event in state["events"]])
                binding=state["events"][0]["result"]
                self.assertEqual("ready",binding["status"])
                self.assertEqual("program_task_binding",binding["source"])
                self.assertIn("no initial model review",binding["summary"])
                record=json.loads((Path(binding["turn_directory"])/"task-binding.json").read_text())
                self.assertEqual(criteria["criteria"],record["criteria"])
                lock=json.loads((project/"criteria.lock.json").read_text())
                self.assertEqual(cli.digest(project/"task.md"),lock["task_sha256"])
                self.assertEqual(cli.digest(project/"criteria.json"),lock["criteria_sha256"])
                self.assertIn("TASK:\n"+self.original,author.prompts[0])
                if adjustable:
                    self.assertIn("Define meaningful initial simulation tasks yourself",author.prompts[0])
                    self.assertIn("There is no initial, preflight or final model review",author.prompts[0])
                    self.assertIn("Only later changes to simulation tasks receive time-limited advice",author.prompts[0])
                    self.assertIn("you decide from that advice while all code gates remain binding",author.prompts[0])
                else:
                    self.assertEqual("free",state["request"]["mode"])
                    self.assertIn("create task_template.py",author.prompts[0])
                    self.assertIn("independent Harness reviewer",author.prompts[0])
                self.assertEqual("max",workflow.call_args.kwargs["reasoning"])

    def test_changed_bound_criteria_are_rejected_before_resumed_author(self):
        project,_,workflow=self.create("locked",author_status="needs_revision")
        workflow.assert_not_called()
        path=project/"criteria.json"
        criteria=json.loads(path.read_text())
        criteria["criteria"][0]["statement"]="Quietly drop the load requirement."
        path.write_text(json.dumps(criteria),encoding="utf-8")
        author=CapturingAuthor()
        with patch.object(cli,"create_worker",return_value=author),patch.object(cli,"SlimWorkflow") as resumed:
            with self.assertRaisesRegex(ValueError,"Frozen criteria changed"):
                cli.resume_project(self.workspace,project)
        self.assertEqual([],author.prompts)
        resumed.assert_not_called()
        self.assertEqual(self.original,(project/"task.md").read_text())


if __name__=="__main__":unittest.main()
