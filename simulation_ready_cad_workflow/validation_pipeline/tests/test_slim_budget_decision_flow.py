"""Real workflow/physics with controlled author and independent-review fixtures."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import trimesh

from ..mechanism.io import write_json_atomic
from ..cold_start.worker import CodexCliWorker
from ..slim_harness.cli import start
from ..slim_harness.cli import create_project
from ..slim_harness.budget import TierConfig
from ..slim_harness.workflow import CHRONO_PYTHON, SlimWorkflow

PREFIX="simulation_ready_cad_workflow.validation_pipeline.slim_harness."
TASK="Let a cube fall under gravity by at least 10 mm within 0.1 seconds."
ADJUSTED="Let a cube fall under gravity by at least 4 mm within 0.1 seconds."


def criterion(task):
    return {"id":"task","kind":"function","statement":task,
            "source_quote":"at least 4 mm" if "4 mm" in task else "at least 10 mm",
            "verification":"Measure the saved cube y-position displacement."}


class Actors(CodexCliWorker):
    session_id="controlled-actor-fixture"
    def __init__(self,action="adjust_task"):
        self.action=action
        self.decisions=[]
        self.task_reviews=[]

    def run_turn(self,workspace,prompt,directory,timeout_seconds,**kwargs):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        project=next(parent for parent in directory.parents if (parent/"workflow.json").is_file())
        props=kwargs.get("response_schema",{}).get("properties",{})
        task=(project/"task.md").read_text(encoding="utf-8")
        result={"status":"ready","summary":"Controlled actor; this is not model-performance evidence.","issues":[],"evidence_paths":[]}
        if "budget_decision" in props:
            self.decisions.append(prompt)
            result["budget_decision"]={"action":self.action,"reason":"The 10 mm target exceeds this short gravity trial; use the remaining budget to retain a smaller measurable fall.",
                                       "tradeoff":"Retain gravity-driven motion while reducing the displacement target; do not claim the original 10 mm target passed."}
            if self.action=="adjust_task":
                write_json_atomic(Path(workspace)/"task_adjustment.json",{"task_text":ADJUSTED,"reason":result["budget_decision"]["reason"],"criteria":[criterion(ADJUSTED)]})
                scene=json.loads((Path(workspace)/"scene.json").read_text())
                scene["tests"][0]["assertions"][0]["expression"]="span(signal('cube.position.y_m')) >= 0.004"
                write_json_atomic(Path(workspace)/"scene.json",scene)
        elif "task_change_decision" in props:
            result["task_change_decision"]={"action":"accept","reason":"The shorter target retains observable gravity-driven motion."}
        elif "conclusion" in props:
            self.task_reviews.append(kwargs)
            result={"conclusion":"A reduced displacement target is meaningful; original target stays failed.","issues":[],"evidence_paths":[]}
        elif "criteria" in props:
            is_adjustment="needs_revision" in props["status"].get("enum",[])
            if is_adjustment:
                self.task_reviews.append(kwargs)
                task=ADJUSTED
            result={"status":"ready","summary":"Controlled independent criteria review", "criteria":[criterion(task)]}
        elif "observations" in props:
            observation_schema=props["observations"]
            ids=observation_schema["items"]["properties"]["criterion_id"].get("enum",["fall"])
            observations=[] if observation_schema.get("maxItems")==0 else [
                {"criterion_id":cid,"verdict":"met","observed_fact":"Controlled review reads the physical result from this integration fixture.","evidence_paths":[str(project/"task.md")]} for cid in ids]
            result={"status":"ready","summary":"Controlled review fixture", "observations":observations,
                    "test_coverage":[{"criterion_id":"fall","test_ids":["drop"],"appropriate":True,"reason":"Direct gravity-displacement measurement"}],
                    "static_coverage":[],"evidence_requests":[]}
        else:
            design=Path(workspace)
            if not (design/"cube.obj").exists():trimesh.creation.box(extents=[.01,.01,.01]).export(design/"cube.obj")
            write_json_atomic(design/"parts.json",{"parts":[{"name":"cube","body":"cube","category":"machine","source_solid_count":1}],
                                                  "body_sources":{"cube":"cube.obj"}})
            target=.004 if "4 mm" in task else .01
            write_json_atomic(design/"scene.json",{"schema_version":"slim_scene_v1","units":"m","gravity_m_s2":[0,-1,0],
                "bodies":[{"id":"cube","source":"cube.obj","fixed":False}],"joints":[],"authoring_sources":["parts.json"],
                "tests":[{"id":"drop","purpose":"Gravity displacement","duration_s":.1,"inputs":[],
                    "assertions":[{"id":"distance","criterion_ids":["task"],"purpose":"Measure declared fall target",
                                   "expression":f"span(signal('cube.position.y_m')) >= {target}"}]}]})
        write_json_atomic(directory/"response.json",result)
        (directory/"prompt.txt").write_text(prompt,encoding="utf-8")
        return result


class BudgetDecisionFlowTests(unittest.TestCase):
    def setUp(self):
        base=Path(__file__).resolve().parents[3]/".runtime_runs/four_gaps_pair_20260912/decision_flow"
        base.mkdir(parents=True,exist_ok=True)
        self.workspace=Path(tempfile.mkdtemp(dir=base))
        self.tier=TierConfig("light",90,15,2,0,10)

    @unittest.skipUnless(Path(CHRONO_PYTHON).is_file(),"Verified Chrono unavailable")
    def test_failed_original_target_is_adjusted_and_really_simulated_again(self):
        actors=Actors()
        repo=Path(__file__).resolve().parents[3]
        with patch.dict(os.environ,{"PYTHONPATH":str(repo)}),patch(PREFIX+"cli.create_worker",return_value=actors),patch(PREFIX+"workflow.create_worker",return_value=actors):
            result=start(self.workspace,"decision",TASK,tier=self.tier,task_adjustable=True)
        project=self.workspace/".runtime_runs/projects/decision"
        self.assertEqual("PASS",result["status"],result)
        self.assertEqual("author_decided_adjusted_task",result["acceptance_target"])
        self.assertFalse(result["original_task_pass_claimed"])
        before=json.loads((project/"runs/r01/result.json").read_text())
        self.assertEqual("NEEDS_REVISION",before["status"])
        self.assertFalse(before["issues"][0]["physical_pass"])
        self.assertEqual(ADJUSTED,(project/"task.md").read_text())
        self.assertEqual(TASK,(project/"task_adjustments/a001/task.md").read_text())
        self.assertTrue(actors.decisions)
        self.assertIn("REMAINING ACTIVE SECONDS",actors.decisions[0])
        self.assertTrue(actors.task_reviews[0]["readonly"])
        self.assertTrue(actors.task_reviews[0]["fresh"])
        self.assertTrue(result["tests"][0]["physical_pass"])
        self.assertEqual(2,result["tests"][0]["geometry_audit"]["candidate_solver_samples"])
        write_json_atomic(self.workspace/"integration-evidence.json",{"fixture_only":True,"model_performance_evidence":False,
            "project":str(project),"original_status":before["status"],"adjusted_status":result["status"],
            "native_runs":2,"independent_reviewer_was_controlled_fixture":True,"original_target_pass_claimed":False})

    def test_frozen_benchmark_cannot_enable_adjustable_policy(self):
        with self.assertRaisesRegex(ValueError,"Frozen benchmark"):
            start(self.workspace,"forbidden",TASK,experiment_path=self.workspace/"not-read.json",task_adjustable=True)
        self.assertFalse((self.workspace/".runtime_runs/projects/forbidden").exists())

    def test_single_simulation_timeout_returns_to_author_choice_with_remaining_budget(self):
        project=create_project(self.workspace,"timeout_choice",TASK)
        criteria={"schema_version":"slim_frozen_criteria_v1","task_file":str(project/"task.md"),
                  "task_text":TASK,"criteria":[criterion(TASK)],"reviewer_summary":"fixture"}
        write_json_atomic(project/"criteria.json",criteria)
        state=json.loads((project/"workflow.json").read_text())
        state["phase"]="geometry"
        state["events"]=[{"kind":"criteria","result":{"status":"ready","criteria":criteria["criteria"],"summary":"fixture"}}]
        write_json_atomic(project/"workflow.json",state)
        actors=Actors("finish_partial")
        with patch(PREFIX+"workflow.create_worker",return_value=actors):
            workflow=SlimWorkflow(self.workspace,project,tier=self.tier)
        with patch.object(workflow,"run_revision",return_value={"status":"SIMULATION_TIMEOUT","issues":["controlled timeout"],"pending_tests":["drop"]}) as validation:
            result=workflow.run()
        self.assertEqual("INCOMPLETE_EVIDENCE",result["status"])
        self.assertEqual("finish_partial",result["budget_decision"]["budget_decision"]["action"])
        self.assertGreater(result["budget_timing"]["remaining_active_seconds"],0)
        self.assertIn("REMAINING ACTIVE SECONDS",actors.decisions[0])
        self.assertEqual(1,validation.call_count)
        self.assertEqual("SIMULATION_TIMEOUT",json.loads((project/"runs/r01/result.json").read_text())["status"])

    def test_adjustable_tradeoffs_require_a_finite_project_budget(self):
        with self.assertRaisesRegex(ValueError,"explicit budget tier"):
            start(self.workspace,"unbounded",TASK,task_adjustable=True)
        self.assertFalse((self.workspace/".runtime_runs/projects/unbounded").exists())


if __name__ == "__main__":unittest.main()
