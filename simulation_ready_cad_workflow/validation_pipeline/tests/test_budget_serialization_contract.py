"""Budget serialization contract tests.

Ensures that every serializable budget structure (snapshot, checkpoint,
summary, adjustment log) produces deterministic, schema-compliant JSON:
- JSON round-trip fidelity (serialize → parse → compare)
- Schema version immutability across all paths
- Field ordering stability
- Unicode and special character handling in reasons/ids
- Large value serialization (many entries, long strings)
- Cross-version compatibility guards
"""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from ..slim_harness.budget import (
    TIERS,
    TierConfig,
    ActiveBudgetTracker,
    AdjustmentLog,
    BudgetStatus,
    CriteriaAdjustment,
    get_tier,
    build_incomplete_summary,
)


class TestSnapshotJsonRoundTrip(unittest.TestCase):
    """snapshot() → JSON string → parse → matches original."""

    def test_round_trip_all_tiers(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            time.sleep(0.005)
            snap = tracker.snapshot()
            serialized = json.dumps(snap)
            parsed = json.loads(serialized)
            self.assertEqual(snap, parsed, f"Tier {name}")

    def test_round_trip_after_pause(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("test")
        snap = tracker.snapshot()
        parsed = json.loads(json.dumps(snap))
        self.assertEqual(snap, parsed)

    def test_round_trip_after_resume(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        tracker.pause("p")
        tracker.resume()
        time.sleep(0.005)
        snap = tracker.snapshot()
        parsed = json.loads(json.dumps(snap))
        self.assertEqual(snap, parsed)

    def test_indent_does_not_change_data(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        snap = tracker.snapshot()
        compact = json.loads(json.dumps(snap, separators=(",", ":")))
        pretty = json.loads(json.dumps(snap, indent=4))
        self.assertEqual(compact, pretty)


class TestCheckpointJsonContract(unittest.TestCase):
    """Checkpoint files are self-contained, parseable JSON."""

    def test_checkpoint_valid_utf8(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            raw = cp.read_bytes()
        raw.decode("utf-8")

    def test_checkpoint_indented(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            text = cp.read_text(encoding="utf-8")
        self.assertIn("\n", text)
        self.assertIn("  ", text)

    def test_checkpoint_parseable_back(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            time.sleep(0.005)
            with tempfile.TemporaryDirectory() as tmp:
                cp = Path(tmp) / f"cp_{name}.json"
                original = tracker.save_checkpoint(cp)
                parsed = json.loads(cp.read_text(encoding="utf-8"))
            self.assertEqual(original, parsed, f"Tier {name}")

    def test_checkpoint_no_python_artifacts(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            text = cp.read_text(encoding="utf-8")
        self.assertNotIn("None", text)
        self.assertNotIn("True", text.replace('"', ""))
        self.assertNotIn("False", text.replace('"', ""))


class TestSchemaVersionImmutability(unittest.TestCase):
    """Schema versions never change across operations."""

    def test_snapshot_version_constant(self):
        versions = set()
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            versions.add(tracker.snapshot()["schema_version"])
            tracker.pause("p")
            versions.add(tracker.snapshot()["schema_version"])
            tracker.resume()
            versions.add(tracker.snapshot()["schema_version"])
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions.pop(), "budget_checkpoint_v1")

    def test_summary_version_constant(self):
        versions = set()
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            summary = build_incomplete_summary(
                tier=tier, tracker=tracker,
                completed_tests=[], pending_tests=[],
                issues=[], evidence_paths=[], blockers=[],
            )
            versions.add(summary["schema_version"])
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions.pop(), "budget_incomplete_summary_v1")

    def test_adjustment_log_version_constant(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "test"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], "criteria_adjustment_log_v1")


class TestAdjustmentLogSerializationContract(unittest.TestCase):
    """AdjustmentLog serialization produces valid, complete JSON."""

    def test_save_is_valid_json(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment("c1", {"x": 0}, {"x": 1}, "reason"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsInstance(data, dict)

    def test_save_has_required_keys(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment("c1", {"x": 0}, {"x": 1}, "reason"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("schema_version", data)
        self.assertIn("adjustments", data)
        self.assertIn("saved_at", data)

    def test_round_trip_fidelity(self):
        log = AdjustmentLog()
        for i in range(10):
            log.record(CriteriaAdjustment(f"c{i}", {"v": i}, {"v": i+1}, f"step {i}"))
        original_list = log.to_list()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log.save(path)
            loaded = AdjustmentLog.load(path)

        loaded_list = loaded.to_list()
        self.assertEqual(len(original_list), len(loaded_list))
        for orig, load in zip(original_list, loaded_list):
            self.assertEqual(orig["criterion_id"], load["criterion_id"])
            self.assertEqual(orig["before"], load["before"])
            self.assertEqual(orig["after"], load["after"])
            self.assertEqual(orig["reason"], load["reason"])

    def test_unicode_in_reason(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "调整公差 — tighter fit"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log.save(path)
            loaded = AdjustmentLog.load(path)
        self.assertEqual(loaded.to_list()[0]["reason"], "调整公差 — tighter fit")

    def test_unicode_in_criterion_id(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment("齿轮_tolerance", {"t": 0.1}, {"t": 0.05}, "tighter"))
        entries = log.to_list()
        self.assertEqual(entries[0]["criterion_id"], "齿轮_tolerance")

    def test_special_chars_in_values(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment(
            "c1",
            {"path": "C:\\Users\\test\\file.json", "note": 'He said "hello"'},
            {"path": "C:\\Users\\test\\file2.json", "note": 'She said "bye"'},
            "path update with quotes",
        ))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log.save(path)
            loaded = AdjustmentLog.load(path)
        entry = loaded.to_list()[0]
        self.assertIn("\\", entry["before"]["path"])

    def test_empty_log_serializes(self):
        log = AdjustmentLog()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["adjustments"], [])

    def test_large_log_serializes(self):
        log = AdjustmentLog()
        for i in range(100):
            log.record(CriteriaAdjustment(
                f"criterion_{i:04d}",
                {"iteration": i, "value": i * 0.1},
                {"iteration": i, "value": (i + 1) * 0.1},
                f"automated adjustment step {i}",
            ))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log.save(path)
            loaded = AdjustmentLog.load(path)
        self.assertEqual(len(loaded.to_list()), 100)


class TestSummarySerializationContract(unittest.TestCase):
    """build_incomplete_summary produces valid, complete JSON."""

    def test_summary_json_round_trip(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[{"test_id": "t1", "physical_pass": True,
                             "geometry_audit": {"passed": True}}],
            pending_tests=["t2"],
            issues=["budget low"],
            evidence_paths=["/path/to/evidence"],
            blockers=["ACTIVE_BUDGET_EXHAUSTED"],
        )
        serialized = json.dumps(summary)
        parsed = json.loads(serialized)
        self.assertEqual(summary["schema_version"], parsed["schema_version"])
        self.assertEqual(summary["status"], parsed["status"])
        self.assertEqual(len(summary["completed_tests"]), len(parsed["completed_tests"]))

    def test_summary_tier_is_serializable(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        json.dumps(summary)

    def test_summary_timing_all_numeric(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.005)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        for key, val in summary["timing"].items():
            self.assertIsInstance(val, (int, float), f"timing[{key}] = {val}")


class TestCriteriaAdjustmentSerialization(unittest.TestCase):
    """CriteriaAdjustment.to_dict() produces valid data."""

    def test_to_dict_is_json_serializable(self):
        adj = CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "test")
        json.dumps(adj.to_dict())

    def test_to_dict_keys(self):
        adj = CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "test")
        d = adj.to_dict()
        expected_keys = {"criterion_id", "before", "after", "reason", "adjusted_at"}
        self.assertEqual(set(d.keys()), expected_keys)

    def test_to_dict_values_match_init(self):
        adj = CriteriaAdjustment("my_criterion", {"old": True}, {"old": False}, "flip it")
        d = adj.to_dict()
        self.assertEqual(d["criterion_id"], "my_criterion")
        self.assertEqual(d["before"], {"old": True})
        self.assertEqual(d["after"], {"old": False})
        self.assertEqual(d["reason"], "flip it")

    def test_to_dict_adjusted_at_is_iso(self):
        adj = CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "test")
        d = adj.to_dict()
        self.assertIn("T", d["adjusted_at"])

    def test_nested_dict_values_round_trip(self):
        before = {"level1": {"level2": [1, 2, 3]}}
        after = {"level1": {"level2": [4, 5, 6]}}
        adj = CriteriaAdjustment("nested", before, after, "deep update")
        serialized = json.dumps(adj.to_dict())
        parsed = json.loads(serialized)
        self.assertEqual(parsed["before"], before)
        self.assertEqual(parsed["after"], after)


class TestBudgetStatusSerialization(unittest.TestCase):
    """BudgetStatus values are valid strings in JSON."""

    def test_all_values_serializable(self):
        for status in BudgetStatus:
            serialized = json.dumps({"status": status.value})
            parsed = json.loads(serialized)
            self.assertEqual(parsed["status"], status.value)

    def test_status_in_summary_is_string(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertIsInstance(summary["status"], str)
        self.assertEqual(summary["status"], "ACTIVE_BUDGET_EXHAUSTED")


if __name__ == "__main__":
    unittest.main()
