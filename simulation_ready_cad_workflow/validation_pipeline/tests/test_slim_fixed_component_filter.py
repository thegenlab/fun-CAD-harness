"""Actual D topology and release/capacity tests for native family filtering."""
import unittest
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.fixed_component_filter import (
    ALL_FAMILIES_MASK, build_fixed_component_filter, component_plan,
)


class Model:
    def __init__(self):
        self.family = 0; self.mask = ALL_FAMILIES_MASK; self.writes = 0
    def GetFamily(self): return self.family
    def GetFamilyMask(self): return self.mask
    def SetFamily(self, value): self.family = value; self.writes += 1
    def SetFamilyMask(self, value): self.mask = value; self.writes += 1


class Body:
    def __init__(self): self.model = Model()
    def GetCollisionModel(self): return self.model


class Link:
    def __init__(self): self.disabled = False
    def IsDisabled(self): return self.disabled


def joint(identifier, first, second, kind="fixed"):
    return {"spec": {"id": identifier, "type": kind, "body_a": first, "body_b": second}, "link": Link(), "active": True}


def allows(plan, first, second):
    return bool(plan["masks"][first] & (1 << plan["families"][second]) and plan["masks"][second] & (1 << plan["families"][first]))


class FixedComponentTests(unittest.TestCase):
    def test_actual_d_r11_omits_only_fixed_component_internal_pairs(self):
        # Frozen D r11 scene graph, 2026-09-08. This fixture has no runtime-file dependency.
        ids = ["frame", "feet", "workbench", "time_barrel", "time_winding_arbor", "time_mainspring",
               "time_click", "time_click_leaf", "alarm_barrel", "alarm_winding_arbor", "alarm_mainspring",
               "alarm_click", "alarm_click_leaf", "center_arbor", "third_arbor", "escape_arbor",
               "pendulum_anchor", "pendulum_bob", "cannon_minute", "cannon_clutch_spring", "motion_intermediate",
               "hour_sleeve", "alarm_setting_ring", "alarm_ring_brake", "coincidence_collar", "trigger_wave_spring",
               "alarm_first_arbor", "alarm_transfer_arbor", "alarm_escape_arbor", "alarm_hammer", "trigger_fork",
               "alarm_sear", "alarm_stops", "alarm_off_slider", "switch_detent_spring", "left_bell", "right_bell", "dial", "dial_marks"]
        edges = [("feet_mount", "frame", "feet"), ("time_spring_root", "time_winding_arbor", "time_mainspring"),
                 ("time_click_leaf_root", "frame", "time_click_leaf"), ("alarm_spring_root", "alarm_winding_arbor", "alarm_mainspring"),
                 ("alarm_click_leaf_root", "frame", "alarm_click_leaf"), ("bob_clamp", "pendulum_anchor", "pendulum_bob"),
                 ("clutch_spring_root", "cannon_minute", "cannon_clutch_spring"), ("alarm_brake_mount", "frame", "alarm_ring_brake"),
                 ("trigger_spring_seat", "hour_sleeve", "trigger_wave_spring"), ("alarm_stops_mount", "frame", "alarm_stops"),
                 ("switch_detent_root", "frame", "switch_detent_spring"), ("left_bell_crown", "frame", "left_bell"),
                 ("right_bell_crown", "frame", "right_bell"), ("dial_mount", "frame", "dial"), ("dial_marks_mount", "dial", "dial_marks")]
        plan = component_plan(ids, edges)
        self.assertEqual(len(ids), 39); self.assertEqual(len(edges), 15)
        self.assertEqual(sorted(map(len, plan["components"])), [2, 2, 2, 2, 2, 11])
        self.assertEqual(plan["omitted_internal_body_pairs"], 60)
        connected = {frozenset((a, b)) for group in plan["components"] for a in group for b in group if a != b}
        for index, first in enumerate(ids):
            for second in ids[index+1:]:
                self.assertEqual(allows(plan, first, second), frozenset((first, second)) not in connected, (first, second))
        for bell in ("left_bell", "right_bell"):
            self.assertTrue(allows(plan, "alarm_hammer", bell))

    def test_release_restores_cross_pairs_and_unchanged_steps_do_not_write(self):
        bodies = {key: Body() for key in "abcd"}
        joints = {"ab": joint("ab", "a", "b"), "bc": joint("bc", "b", "c"), "cd": joint("cd", "c", "d", "revolute")}
        handle = build_fixed_component_filter(bodies, joints)
        writes = sum(body.model.writes for body in bodies.values())
        self.assertFalse(handle.refresh(time_s=.001)); self.assertEqual(writes, sum(body.model.writes for body in bodies.values()))
        self.assertFalse(allows(handle.plan, "a", "c")); self.assertTrue(allows(handle.plan, "c", "d"))
        joints["ab"]["active"] = False; joints["ab"]["link"].disabled = True
        self.assertTrue(handle.refresh(time_s=.002))
        self.assertTrue(allows(handle.plan, "a", "b")); self.assertTrue(allows(handle.plan, "a", "c"))
        self.assertFalse(allows(handle.plan, "b", "c")); self.assertEqual(bodies["a"].model.mask, ALL_FAMILIES_MASK)

    def test_an_alternate_live_fixed_path_keeps_the_component_together(self):
        bodies = {key: Body() for key in "abc"}
        joints = {"ab": joint("ab", "a", "b"), "bc": joint("bc", "b", "c"), "ac": joint("ac", "a", "c")}
        handle = build_fixed_component_filter(bodies, joints); before = handle.native_model_updates
        joints["ab"]["link"].disabled = True
        self.assertTrue(handle.refresh(time_s=.001)); self.assertFalse(allows(handle.plan, "a", "b"))
        self.assertEqual(handle.native_model_updates, before)

    def test_native_family_capacity_restores_all_default_collisions(self):
        ids = [f"body_{i:02d}" for i in range(30)]
        edges = [(f"fixed_{i}", ids[2*i], ids[2*i+1]) for i in range(15)]
        plan = component_plan(ids, edges)
        self.assertFalse(plan["capacity_available"]); self.assertEqual(plan["omitted_internal_body_pairs"], 0)
        self.assertEqual(set(plan["families"].values()), {0}); self.assertEqual(set(plan["masks"].values()), {ALL_FAMILIES_MASK})


if __name__ == "__main__": unittest.main()
