"""CriteriaAdjustment and AdjustmentLog edge case tests.

Tests unusual inputs, unicode handling, long strings, empty logs,
repeated appends, and save/load edge cases for the criteria
adjustment tracking subsystem.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ..slim_harness.budget import (
    AdjustmentLog,
    CriteriaAdjustment,
)


class TestCriteriaAdjustmentEdges(unittest.TestCase):
    """Edge case construction and validation."""

    def test_unicode_criterion_id(self):
        adj = CriteriaAdjustment(
            criterion_id="标准_α",
            before={"v": 1}, after={"v": 2},
            reason="International criterion",
            frozen=False,
        )
        self.assertEqual(adj.criterion_id, "标准_α")

    def test_long_reason(self):
        reason = "x" * 10000
        adj = CriteriaAdjustment(
            criterion_id="c1",
            before={"v": 1}, after={"v": 2},
            reason=reason,
            frozen=False,
        )
        self.assertEqual(len(adj.reason), 10000)

    def test_nested_before_after(self):
        adj = CriteriaAdjustment(
            criterion_id="c1",
            before={"a": {"b": {"c": 1}}},
            after={"a": {"b": {"c": 2}}},
            reason="Deep change",
            frozen=False,
        )
        d = adj.to_dict()
        self.assertEqual(d["before"]["a"]["b"]["c"], 1)
        self.assertEqual(d["after"]["a"]["b"]["c"], 2)

    def test_frozen_true_raises(self):
        with self.assertRaises(ValueError):
            CriteriaAdjustment(
                criterion_id="frozen_crit",
                before={"v": 1}, after={"v": 2},
                reason="Should fail",
                frozen=True,
            )

    def test_same_before_after_raises(self):
        with self.assertRaises(ValueError):
            CriteriaAdjustment(
                criterion_id="c1",
                before={"v": 5}, after={"v": 5},
                reason="No change",
                frozen=False,
            )

    def test_empty_reason_raises(self):
        with self.assertRaises(ValueError):
            CriteriaAdjustment(
                criterion_id="c1",
                before={"v": 1}, after={"v": 2},
                reason="",
                frozen=False,
            )


class TestCriteriaAdjustmentToDict(unittest.TestCase):
    """to_dict output keys and values."""

    def _adj(self):
        return CriteriaAdjustment(
            criterion_id="c1",
            before={"val": 10}, after={"val": 20},
            reason="Updated threshold",
            frozen=False,
        )

    def test_has_criterion_id_key(self):
        d = self._adj().to_dict()
        self.assertIn("criterion_id", d)
        self.assertEqual(d["criterion_id"], "c1")

    def test_has_before_after(self):
        d = self._adj().to_dict()
        self.assertIn("before", d)
        self.assertIn("after", d)

    def test_has_reason(self):
        d = self._adj().to_dict()
        self.assertEqual(d["reason"], "Updated threshold")

    def test_has_adjusted_at(self):
        d = self._adj().to_dict()
        self.assertIn("adjusted_at", d)
        self.assertIsInstance(d["adjusted_at"], str)


class TestAdjustmentLogEdges(unittest.TestCase):
    """AdjustmentLog with unusual patterns."""

    def _adj(self, cid="c1", val=1):
        return CriteriaAdjustment(
            criterion_id=cid,
            before={"v": val}, after={"v": val + 100},
            reason=f"Adjust {cid}",
            frozen=False,
        )

    def test_empty_log(self):
        log = AdjustmentLog()
        self.assertEqual(len(log.entries), 0)

    def test_single_entry(self):
        log = AdjustmentLog()
        log.record(self._adj())
        self.assertEqual(len(log.entries), 1)

    def test_order_preserved(self):
        log = AdjustmentLog()
        for i in range(5):
            log.record(self._adj(f"c{i}", i))
        ids = [e.criterion_id for e in log.entries]
        self.assertEqual(ids, [f"c{i}" for i in range(5)])

    def test_save_empty_log(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "empty.json"
            log = AdjustmentLog()
            log.save(p)
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(data["schema_version"], "criteria_adjustment_log_v1")
            self.assertEqual(len(data["adjustments"]), 0)

    def test_load_empty_log(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "empty.json"
            log = AdjustmentLog()
            log.save(p)
            loaded = AdjustmentLog.load(p)
            self.assertEqual(len(loaded.entries), 0)

    def test_large_log(self):
        log = AdjustmentLog()
        for i in range(100):
            log.record(self._adj(f"c{i}", i))
        self.assertEqual(len(log.entries), 100)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "large.json"
            log.save(p)
            loaded = AdjustmentLog.load(p)
            self.assertEqual(len(loaded.entries), 100)

    def test_unicode_in_log(self):
        adj = CriteriaAdjustment(
            criterion_id="日本語",
            before={"v": "旧"}, after={"v": "新"},
            reason="国際化テスト",
            frozen=False,
        )
        log = AdjustmentLog()
        log.record(adj)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "unicode.json"
            log.save(p)
            loaded = AdjustmentLog.load(p)
            self.assertEqual(loaded.entries[0].criterion_id, "日本語")

    def test_load_nonexistent_raises(self):
        with self.assertRaises(Exception):
            AdjustmentLog.load(Path("/nonexistent/path.json"))

    def test_load_malformed_json_raises(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.json"
            p.write_text("not json", encoding="utf-8")
            with self.assertRaises(Exception):
                AdjustmentLog.load(p)


class TestAdjustmentLogSchemaVersion(unittest.TestCase):
    """Saved logs have correct schema version."""

    def test_schema_version(self):
        log = AdjustmentLog()
        adj = CriteriaAdjustment(
            criterion_id="c1",
            before={"v": 1}, after={"v": 2},
            reason="Test",
            frozen=False,
        )
        log.record(adj)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "log.json"
            log.save(p)
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(data["schema_version"], "criteria_adjustment_log_v1")

    def test_schema_version_stable_across_saves(self):
        log = AdjustmentLog()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "log.json"
            log.save(p)
            v1 = json.loads(p.read_text(encoding="utf-8"))["schema_version"]
            adj = CriteriaAdjustment(
                criterion_id="c1",
                before={"v": 1}, after={"v": 2},
                reason="Test",
                frozen=False,
            )
            log.record(adj)
            log.save(p)
            v2 = json.loads(p.read_text(encoding="utf-8"))["schema_version"]
            self.assertEqual(v1, v2)


if __name__ == "__main__":
    unittest.main()
