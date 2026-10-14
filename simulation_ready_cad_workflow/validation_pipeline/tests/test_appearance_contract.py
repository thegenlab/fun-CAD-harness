from __future__ import annotations

import unittest

from simulation_ready_cad_workflow.validation_pipeline.contracts.appearance import (
    APPEARANCE_PLAN_SCHEMA,
    AppearancePlanError,
    STATIC_VISUAL_GATE_VERSION,
    required_visual_roles,
    validate_appearance_plan,
)


def _plan() -> dict:
    return {
        "schema_version": APPEARANCE_PLAN_SCHEMA,
        "product_identity": {
            "recognizable_as": "calibrated hanging spring scale",
            "must_show": ["graduated scale", "pointer", "curved load hook"],
            "must_not_look_like": ["stack of anonymous slabs and bars"],
        },
        "components": [
            {
                "component_id": "carriage",
                "name": "guided pointer carriage",
                "owner_rigid_group": "moving_group",
                "visible_features": ["pointer", "guide engagement"],
                "interfaces": ["spring anchor", "load hook stem"],
                "positive_checks": ["pointer aligns with the scale"],
                "negative_checks": ["not an unexplained rectangular block"],
            }
        ],
        "view_requirements": [
            {
                "role": "guide_section",
                "kind": "section",
                "purpose": "show carriage capture and guide clearance",
                "elevation_deg": 20.0,
                "azimuth_deg": -45.0,
                "section_axis": "y",
                "section_position_mm": 0.0,
                "section_keep": "below",
            },
            {
                "role": "loaded_pose",
                "kind": "pose",
                "purpose": "show loaded pointer travel",
                "elevation_deg": 20.0,
                "azimuth_deg": -45.0,
                "source_step": "review_poses/loaded.step",
            },
        ],
    }


class AppearanceContractTests(unittest.TestCase):
    def test_normalizes_component_inventory_and_required_views(self) -> None:
        plan = validate_appearance_plan(_plan())
        roles = required_visual_roles(
            {
                "appearance_plan": plan,
                "motion_review_roles": ["loaded_pose"],
            },
            gate_version=STATIC_VISUAL_GATE_VERSION,
        )

        self.assertEqual(plan["components"][0]["component_id"], "carriage")
        self.assertEqual(
            roles,
            (
                "front",
                "rear",
                "left",
                "right",
                "top",
                "bottom",
                "iso",
                "guide_section",
                "loaded_pose",
            ),
        )

    def test_rejects_missing_negative_proxy_checks(self) -> None:
        value = _plan()
        value["components"][0]["negative_checks"] = []

        with self.assertRaises(AppearancePlanError):
            validate_appearance_plan(value)

    def test_pose_view_requires_a_separate_step_path(self) -> None:
        value = _plan()
        value["view_requirements"][1].pop("source_step")

        with self.assertRaises(AppearancePlanError):
            validate_appearance_plan(value)


if __name__ == "__main__":
    unittest.main()
