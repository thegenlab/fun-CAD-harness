"""原声明环境导向恢复，不放宽正式实体连接校验。"""

from pathlib import Path
import copy, json, tempfile, unittest
from benchmark_templates.fixtures import prepared_with_source_fixture_joint


class SourceFixtureJointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def data(self):
        scene = {
            "schema_version": "slim_scene_v1",
            "units": "mm",
            "bodies": [
                {"id": "frame", "source": "frame.step", "fixed": True},
                {"id": "load", "source": "load.step", "fixed": False},
            ],
            "joints": [
                {
                    "id": "load_guide",
                    "type": "prismatic",
                    "body_a": "frame",
                    "body_b": "load",
                    "anchor": [0, 0, -280],
                    "axis": [0, 0, 2],
                    "damping": 12.0,
                    "description": "原测试环境防摆约束，非打印导轨",
                    "region_a": {"points": [[0, 0, -80], [0, 0, -360]], "radius": 20},
                    "region_b": {"points": [[0, 0, -235], [0, 0, -325]], "radius": 45},
                }
            ],
            "gravity_m_s2": [0, 0, -9.81],
            "tests": [],
        }
        prepared = {
            "units": "m",
            "bodies": [
                {"id": "frame", "position_m": [0.01, 0, -0.02], "mass_kg": 2},
                {"id": "load", "position_m": [0, 0, -0.28], "mass_kg": 5.69},
            ],
            "joints": [],
            "gravity_m_s2": [0, 0, -9.81],
            "cad_loader": {"origin": {"untouched": "same source view"}},
        }
        source = self.root / "scene.json"
        path = self.root / "prepared.json"
        source.write_text(json.dumps(scene))
        path.write_text(json.dumps(prepared))
        return scene, prepared, source, path

    def test_restore_only_selected_original_prismatic_with_units_and_damping(self):
        scene, prepared, source, path = self.data()
        before_source = source.read_bytes()
        before_prepared = path.read_bytes()
        result = prepared_with_source_fixture_joint(
            path, source, "load_guide", purpose="源操作说明明确负载环境防摆"
        )
        joint = result["joints"][0]
        self.assertEqual(joint["anchor"], [0.0, 0.0, -0.28])
        self.assertEqual(joint["axis"], [0.0, 0.0, 1.0])
        self.assertEqual(joint["damping"], 12.0)
        self.assertTrue(joint["source_declared_exact"])
        self.assertEqual(
            joint["region_a"]["points"],
            [[-0.01, 0.0, -0.06], [-0.01, 0.0, -0.33999999999999997]],
        )
        self.assertAlmostEqual(joint["region_b"]["points"][0][2], 0.045)
        self.assertEqual(
            result["cad_loader"]["source_fixture_joints"][0]["original_spec"],
            scene["joints"][0],
        )
        restored = copy.deepcopy(result)
        restored["joints"].pop()
        restored["cad_loader"].pop("source_fixture_joints")
        self.assertEqual(restored, prepared)
        self.assertEqual(source.read_bytes(), before_source)
        self.assertEqual(path.read_bytes(), before_prepared)

    def test_duplicate_or_nonprismatic_is_not_silently_reinterpreted(self):
        scene, prepared, source, path = self.data()
        with self.assertRaisesRegex(ValueError, "用途"):
            prepared_with_source_fixture_joint(path, source, "load_guide", purpose="")
        scene["joints"][0]["type"] = "revolute"
        source.write_text(json.dumps(scene))
        with self.assertRaisesRegex(ValueError, "prismatic"):
            prepared_with_source_fixture_joint(
                path, source, "load_guide", purpose="not an allowed environment guide"
            )
        scene["joints"][0]["type"] = "prismatic"
        source.write_text(json.dumps(scene))
        prepared["joints"] = [{"id": "load_guide"}]
        path.write_text(json.dumps(prepared))
        with self.assertRaisesRegex(ValueError, "重复"):
            prepared_with_source_fixture_joint(
                path, source, "load_guide", purpose="explicit source guide"
            )


if __name__ == "__main__":
    unittest.main()
