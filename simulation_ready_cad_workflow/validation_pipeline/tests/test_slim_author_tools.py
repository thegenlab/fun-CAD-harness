"""Author-tool integration; test fixtures are not benchmark certifications."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ..slim_harness.asset import build_file
from ..slim_harness.author_tools import run_tool, project_for_scene
from ..slim_harness.cli import create_project, copy_delivery
from ..slim_harness.frozen_experiment import install_experiment
from ..slim_harness.geometry import prepare_geometry
from ..slim_harness.trace_storage import file_digest
from ..slim_harness.measurement_registry import value_digest


class AuthorToolTests(unittest.TestCase):
    def setUp(self):
        self.workspace=Path(__file__).resolve().parents[3]
        root=self.workspace/".runtime_runs"/"author_tool_tests";root.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def model(self,with_tests=False):
        source=self.root/"model.py"
        source.write_text("""from build123d import Box, Pos
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.asset import Asset, Part
def build():
    return Asset(parts={
        'base':Part(Box(40,30,4),body='frame',role='fixed'),
        'foot':Part(Pos(22,0,0)*Box(4,4,4),body='frame',role='fixed'),  # touches the base edge: one welded frame
        'weight':Part(Pos(0,0,12)*Box(4,4,4),body='moving')},
        gravity_m_s2=[0,0,-9.81], tests=TESTS)
""".replace("TESTS",repr([{"id":"fall","purpose":"Free fall fixture","duration_s":.005,
            "assertions":[{"id":"motion","purpose":"Downward motion","expression":"last(signal('moving.position.z_m')) < first(signal('moving.position.z_m'))"}]}] if with_tests else [])),encoding="utf-8")
        return source

    def test_named_parts_export_bodies_without_manual_solid_mapping(self):
        result=build_file(self.model(),self.root/"generated")
        scene=json.loads(Path(result["scene"]).read_text())
        self.assertEqual(["frame","moving"],[b["id"] for b in scene["bodies"]])
        self.assertTrue(all("solid_ids" not in b for b in scene["bodies"]))
        geometry=prepare_geometry(Path(result["scene"]),self.root/"geometry")
        self.assertEqual([],geometry["errors"])
        self.assertEqual([2,1],[len(b["source_solid_ids"]) for b in geometry["bodies"]])
        self.assertEqual(["base","foot","weight"],[p["name"] for p in json.loads((self.root/"generated/parts.json").read_text())["parts"]])

    def test_preview_needs_no_tests_and_does_not_modify_source(self):
        result=build_file(self.model(),self.root/"generated")
        scene=Path(result["scene"]);original=scene.read_bytes()
        with patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.author_tools.render_views",return_value=self.root/"preview.png"):
            preview=run_tool("preview",scene,workspace=self.workspace)
        self.assertEqual("PREVIEW_READY",preview["status"])
        self.assertFalse(preview["certified"])
        self.assertEqual(original,scene.read_bytes())
        checked=run_tool("check",scene,workspace=self.workspace)
        self.assertEqual("DEVELOPMENT_FAILURE",checked["status"])
        self.assertTrue(any("test" in issue.lower() for issue in checked["issues"]))

    def test_check_uses_source_geometry_and_keeps_formal_result_separate(self):
        result=build_file(self.model(with_tests=True),self.root/"generated")
        checked=run_tool("check",result["scene"],workspace=self.workspace)
        self.assertEqual("CHECKED",checked["status"])
        self.assertFalse(checked["certified"])
        self.assertEqual(["fall"],checked["tests"])
        self.assertEqual("slim_author_tool_result_v1",json.loads((Path(checked["attempt"])/"result.json").read_text())["schema_version"])

    def test_nested_helpers_survive_normal_submit_copy(self):
        result=build_file(self.model(with_tests=True),self.root/"generated")
        source=Path(result["scene"])
        (source.parent/"helpers").mkdir();(source.parent/"helpers/parameters.py").write_text("LENGTH=4\n")
        scene=json.loads(source.read_text());scene["authoring_sources"].append("helpers/parameters.py")
        source.write_text(json.dumps(scene))
        target=self.root/"submitted";target.mkdir()
        copy_delivery(source,target)
        self.assertTrue((target/"helpers/parameters.py").is_file())
        self.assertIn("helpers/parameters.py",json.loads((target/"scene.json").read_text())["authoring_sources"])

    def test_project_inference_and_frozen_development_tests(self):
        workspace=self.root/"workspace";workspace.mkdir()
        project=create_project(workspace,"frozen","A weight must fall.")
        design=project/"runs/r01/design"
        result=build_file(self.model(),design)
        scene=Path(result["scene"])
        experiment={"schema_version":"slim_frozen_experiment_v1","units":"mm","task_text":"A weight must fall.",
            "reviewer_summary":"Researcher fixture","criteria":[{"id":"fall","kind":"function","statement":"Fall",
            "source_quote":"must fall","verification":"Position"}],"gravity_m_s2":[0,0,-9.81],
            "tests":[{"id":"fall","purpose":"Free fall","duration_s":.005,"assertions":[{"id":"m","purpose":"Motion",
                "expression":"last(signal('moving.position.z_m')) < first(signal('moving.position.z_m'))"}]}]}
        path=self.root/"frozen.json";path.write_text(json.dumps(experiment))
        state=json.loads((project/"workflow.json").read_text())
        state["request"]["frozen_experiment_sha256"]=install_experiment(project,path)
        (project/"workflow.json").write_text(json.dumps(state))
        self.assertEqual(project,project_for_scene(scene,workspace))
        checked=run_tool("check",scene,workspace=workspace)
        self.assertEqual("CHECKED",checked["status"],checked)
        self.assertNotIn("tests",json.loads(scene.read_text()))
        # Isolate registration plumbing from the clock-specific measurement
        # kernel. The loader still checks real tests, criteria and plan files.
        (project/"criteria.json").write_text(json.dumps({"criteria":experiment["criteria"]}))
        (design/"plan.json").write_text('{"fixture":"reviewed source plan"}')
        registry={"schema_version":"slim_project_measurement_registration_v1",
            "criteria_sha256":file_digest(project/"criteria.json"),"task_sha256":file_digest(project/"task.md")}
        binding={"test_id":"fall","criteria_sha256":registry["criteria_sha256"],"task_sha256":registry["task_sha256"],
            "source_test_spec_sha256":value_digest(experiment["tests"][0]),"source_plan_bindings":{"plan.json":file_digest(design/"plan.json")}}
        registry["bindings"]=[binding]
        (project/"measurement-registration.json").write_text(json.dumps(registry))
        with patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.measurement_registry.validate_runtime_registration"):
            registered=run_tool("check",scene,workspace=workspace)
        self.assertEqual("CHECKED",registered["status"],registered)
        self.assertTrue((Path(registered["attempt"])/"design/plan.json").is_file())
        changed=json.loads(scene.read_text());changed["tests"]=experiment["tests"]
        changed["tests"][0]["duration_s"]=.001;scene.write_text(json.dumps(changed))
        refused=run_tool("check",scene,workspace=workspace)
        self.assertEqual("DEVELOPMENT_FAILURE",refused["status"])
        self.assertIn("cannot be changed",refused["issues"][0])

    def test_external_build_cannot_overwrite_frozen_destination_tests(self):
        workspace=self.root/"workspace";workspace.mkdir()
        project=create_project(workspace,"frozen","A weight must fall.")
        destination=project/"runs/r01/design"
        original=b'{"preserved":"previous candidate"}'
        (destination/"scene.json").write_bytes(original)
        experiment={"schema_version":"slim_frozen_experiment_v1","units":"mm","task_text":"A weight must fall.",
            "reviewer_summary":"Researcher fixture","criteria":[{"id":"fall","kind":"function","statement":"Fall",
            "source_quote":"must fall","verification":"Position"}],"gravity_m_s2":[0,0,-9.81],
            "tests":[{"id":"fall","purpose":"Frozen test","duration_s":.01,
                      "assertions":[{"id":"m","purpose":"Motion","expression":"last(signal('time_s')) >= 0.01"}]}]}
        path=self.root/"frozen.json";path.write_text(json.dumps(experiment))
        state=json.loads((project/"workflow.json").read_text())
        state["request"]["frozen_experiment_sha256"]=install_experiment(project,path)
        (project/"workflow.json").write_text(json.dumps(state))
        prefix="simulation_ready_cad_workflow.validation_pipeline.slim_harness.author_tools._build"
        def build(source,output,attempt,workspace,*,project=None,cad_python=None):return Path(build_file(source,output)["scene"])
        publisher="simulation_ready_cad_workflow.validation_pipeline.slim_harness.author_tools.publish_attempt"
        with patch(prefix,side_effect=build), patch(publisher):
            refused=run_tool("build",self.model(with_tests=True),workspace=workspace,output=destination)
        self.assertEqual("DEVELOPMENT_FAILURE",refused["status"],refused)
        self.assertEqual(original,(destination/"scene.json").read_bytes())
        with patch(prefix,side_effect=build), patch(publisher):
            accepted=run_tool("build",self.model(),workspace=workspace,output=destination)
        self.assertEqual("BUILT",accepted["status"],accepted)
        self.assertEqual(experiment["tests"],json.loads((destination/"scene.json").read_text())["tests"])

    def test_failed_build_preserves_original_source(self):
        source=self.root/"broken.py";original="def build():\n    raise RuntimeError('broken fixture')\n"
        source.write_text(original)
        with self.assertRaises(RuntimeError):build_file(source,self.root/"failed")
        source.write_text("# later edit\n")
        self.assertEqual(original,(self.root/"failed/broken.py").read_text())


if __name__=="__main__":unittest.main()
