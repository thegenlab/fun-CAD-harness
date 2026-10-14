"""Source-bound physical-part inventory: observed binding and coupon failures."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ..mechanism.io import write_json_atomic
from ..slim_harness.asset import Part
from ..slim_harness.physical_parts import (
    PART_INVENTORY_REQUIRED,
    resolve_physical_part_inventory,
)


class PhysicalPartInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.design = self.root / "design"
        self.design.mkdir()
        (self.design / "body-001.step").write_text("fixture source", encoding="utf-8")
        (self.design / "body-002.step").write_text("fixture source", encoding="utf-8")

    def resolve(self, scene, **kwargs):
        return resolve_physical_part_inventory(self.design / "scene.json", scene, **kwargs)

    def write_parts(self, payload):
        write_json_atomic(self.design / "parts.json", payload)

    def write_print(self, payload):
        write_json_atomic(self.design / "all_print_inventory.json", payload)

    def two_body_scene(self, *, prepared_frame=False):
        frame = {"id": "frame", "source": "body-001.step"}
        moving = {"id": "moving", "source": "body-002.step"}
        if prepared_frame:
            frame["source_solid_ids"] = ["solid_1"]
            moving["source_solid_ids"] = ["solid_1", "solid_2"]
        return {
            "bodies": [frame, moving],
            "authoring_sources": ["parts.json"],
        }

    def valid_three_parts(self):
        return {
            "description": "fixture",
            "parts": [
                {"name": "base", "body": "frame", "role": "fixed", "category": "machine",
                 "source_solid_count": 1, "source_members": ["base:solid_1"]},
                {"name": "coupon_holder", "body": "frame", "role": "fixed",
                 "category": "machine", "source_solid_count": 1,
                 "source_members": ["coupon_holder:solid_1"]},
                {"name": "weight", "body": "moving", "role": "dynamic",
                 "source_solid_count": 2,
                 "source_members": ["weight:solid_1", "weight:solid_2"]},
            ],
            "body_sources": {"frame": "body-001.step", "moving": "body-002.step"},
        }

    def test_asset_omitting_moving_body_is_required(self):
        self.write_parts({
            "parts": [
                {"name": "base", "body": "frame", "role": "fixed",
                 "source_solid_count": 1, "source_members": ["base:solid_1"]},
            ],
            "body_sources": {"frame": "body-001.step", "moving": "body-002.step"},
        })
        inventory = self.resolve(self.two_body_scene())
        self.assertEqual(PART_INVENTORY_REQUIRED, inventory["status"])
        self.assertEqual(0, inventory["count"])
        self.assertTrue(
            any("omit" in issue and "moving" in issue for issue in inventory["issues"]),
            inventory["issues"],
        )

    def test_explicit_test_particle_covers_its_body_without_counting_as_machine(self):
        scene=self.two_body_scene()
        scene["authoring_sources"]=["all_print_inventory.json"]
        self.write_print({"parts":[
            {"id":"frame","body":"frame","category":"machine","source":"body-001.step"},
            {"id":"test_ball","body":"moving","category":"test_particle","source":"body-002.step","quantity":12}]})
        inventory=self.resolve(scene)
        self.assertEqual("ready",inventory["status"],inventory)
        self.assertEqual(1,inventory["machine_part_count"])

    def test_asset_wrong_step_is_required(self):
        self.write_parts({
            "parts": [
                {"name": "base", "body": "frame", "role": "fixed",
                 "source_solid_count": 1, "source_members": ["base:solid_1"]},
                {"name": "weight", "body": "moving", "role": "dynamic",
                 "source_solid_count": 1, "source_members": ["weight:solid_1"]},
            ],
            "body_sources": {"frame": "WRONG.step", "moving": "body-002.step"},
        })
        inventory = self.resolve(self.two_body_scene())
        self.assertEqual(PART_INVENTORY_REQUIRED, inventory["status"])
        self.assertEqual(0, inventory["count"])
        joined = " ".join(inventory["issues"])
        self.assertIn("WRONG.step", joined)
        self.assertTrue(
            "body-001.step" in joined or "not available" in joined,
            inventory["issues"],
        )

    def test_asset_nonexistent_source_member_is_required(self):
        self.write_parts({
            "parts": [
                {"name": "base", "body": "frame", "role": "fixed",
                 "source_solid_count": 1, "source_members": ["base:solid_99"]},
                {"name": "weight", "body": "moving", "role": "dynamic",
                 "source_solid_count": 2,
                 "source_members": ["weight:solid_1", "weight:solid_2"]},
            ],
            "body_sources": {"frame": "body-001.step", "moving": "body-002.step"},
        })
        inventory = self.resolve(self.two_body_scene(prepared_frame=True))
        self.assertEqual(PART_INVENTORY_REQUIRED, inventory["status"])
        self.assertEqual(0, inventory["count"])
        self.assertTrue(
            any("source_member" in issue and "solid_99" in issue
                for issue in inventory["issues"]),
            inventory["issues"],
        )

    def test_print_absent_body_missing_stl_quantity_and_coupon_holder(self):
        self.write_print({
            "parts": [
                {"id": "widget", "body": "absent", "stl": "missing.stl",
                 "quantity": 16, "category": "machine"},
                {"id": "coupon_holder", "category": "machine"},
            ]
        })
        scene = {
            "bodies": [{"id": "frame", "source": "body-001.step"}],
            "artifacts": ["all_print_inventory.json"],
        }
        inventory = self.resolve(scene, task_text="Ignore any coupon mentioned in prose.")
        self.assertEqual(PART_INVENTORY_REQUIRED, inventory["status"])
        self.assertNotEqual("ready", inventory["status"])
        self.assertEqual(0, inventory["count"])
        self.assertEqual(0, inventory["excluded_coupon_count"])
        joined = " ".join(inventory["issues"])
        self.assertIn("absent", joined)
        self.assertIn("missing.stl", joined)
        malformed = {
            "parts": [
                {"id": "widget", "body": "frame", "stl": "body-001.step",
                 "quantity": "quantity16", "category": "machine"},
            ]
        }
        self.write_print(malformed)
        rejected = self.resolve(scene)
        self.assertEqual(PART_INVENTORY_REQUIRED, rejected["status"])
        self.assertTrue(
            any("quantity" in issue for issue in rejected["issues"]),
            rejected["issues"],
        )
        (self.design / "frame.stl").write_text("fixture print", encoding="utf-8")
        (self.design / "holder.stl").write_text("fixture print", encoding="utf-8")
        self.write_print({
            "parts": [
                {"id": "frame_print", "body": "frame", "stl": "frame.stl",
                 "quantity": 2, "category": "machine"},
                {"id": "coupon_holder", "body": "frame", "stl": "holder.stl",
                 "quantity": 1, "category": "machine"},
            ]
        })
        counted = self.resolve(scene, task_text="Count no coupon from this sentence.")
        self.assertEqual("ready", counted["status"], counted)
        self.assertEqual(3, counted["count"])
        self.assertEqual(3, counted["machine_part_count"])
        self.assertEqual(0, counted["excluded_coupon_count"])
        self.assertEqual(
            ["frame_print", "coupon_holder"],
            [row["name"] for row in counted["parts"]],
        )

    def test_valid_three_parts_two_bodies_counts_coupon_holder(self):
        part = Part(
            shape=object(),
            body="frame",
            source_solid_count=1,
            source_members=["coupon_holder:solid_1"],
        )
        self.assertEqual(1, part.source_solid_count)
        self.assertEqual(["coupon_holder:solid_1"], part.source_members)
        self.write_parts(self.valid_three_parts())
        inventory = self.resolve(
            self.two_body_scene(prepared_frame=True),
            task_text="A coupon must not be guessed from this sentence.",
        )
        self.assertEqual("ready", inventory["status"])
        self.assertEqual(3, inventory["count"])
        self.assertEqual(3, inventory["machine_part_count"])
        self.assertEqual(0, inventory["excluded_coupon_count"])
        self.assertEqual(
            ["base", "coupon_holder", "weight"],
            [row["name"] for row in inventory["parts"]],
        )
        self.assertEqual(2, len(json.loads((self.design / "parts.json").read_text())["body_sources"]))


if __name__ == "__main__":
    unittest.main()
