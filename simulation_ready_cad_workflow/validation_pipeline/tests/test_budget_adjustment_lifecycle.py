"""CriteriaAdjustment and AdjustmentLog lifecycle tests.

Covers save/load round-trips, append-only invariants, multi-entry
ordering, frozen enforcement chains, and schema validation on load.
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


class TestCriteriaAdjustmentConstruction(unittest.TestCase):
    """Basic CriteriaAdjustment construction and validation."""

    def test_valid_adjustment(self):
        adj = CriteriaAdjustment(
            criterion_id="c1",
            before={"tol": 0.1},
            after={"tol": 0.2},
            reason="relaxed tolerance",
        )
        self.assertEqual(adj.criterion_id, "c1")

    def test_frozen_raises(self):
        with self.assertRaises(ValueError):
            CriteriaAdjustment(
                criterion_id="c2",
                before={"a": 1},
                after={"a": 2},
                reason="try",
                frozen=True,
            )

    def test_same_before_after_raises(self):
        with self.assertRaises(ValueError):
            CriteriaAdjustment(
                criterion_id="c3",
                before={"x": 1},
                after={"x": 1},
                reason="noop",
            )

    def test_empty_reason_raises(self):
        with self.assertRaises(ValueError):
            CriteriaAdjustment(
                criterion_id="c4",
                before={"a": 1},
                after={"a": 2},
                reason="",
            )

    def test_auto_timestamp(self):
        adj = CriteriaAdjustment(
            criterion_id="c5",
            before={"v": 1},
            after={"v": 2},
            reason="update",
        )
        self.assertTrue(len(adj.adjusted_at) > 0)

    def test_explicit_timestamp_kept(self):
        adj = CriteriaAdjustment(
            criterion_id="c6",
            before={"v": 1},
            after={"v": 2},
            reason="update",
            adjusted_at="2025-01-01T00:00:00+00:00",
        )
        self.assertEqual(adj.adjusted_at, "2025-01-01T00:00:00+00:00")


class TestCriteriaAdjustmentToDict(unittest.TestCase):
    """to_dict output structure."""

    def test_keys(self):
        adj = CriteriaAdjustment(
            criterion_id="k1",
            before={"a": 1},
            after={"a": 2},
            reason="change",
        )
        d = adj.to_dict()
        self.assertEqual(
            set(d.keys()),
            {"criterion_id", "before", "after", "reason", "adjusted_at"},
        )

    def test_criterion_id_key_not_criterion(self):
        adj = CriteriaAdjustment(
            criterion_id="verify",
            before={"x": 0},
            after={"x": 1},
            reason="fix",
        )
        d = adj.to_dict()
        self.assertIn("criterion_id", d)
        self.assertNotIn("criterion", d)

    def test_values_match(self):
        adj = CriteriaAdjustment(
            criterion_id="v1",
            before={"old": True},
            after={"old": False},
            reason="toggle",
        )
        d = adj.to_dict()
        self.assertEqual(d["before"], {"old": True})
        self.assertEqual(d["after"], {"old": False})
        self.assertEqual(d["reason"], "toggle")

    def test_frozen_field_not_in_dict(self):
        adj = CriteriaAdjustment(
            criterion_id="f1",
            before={"a": 1},
            after={"a": 2},
            reason="adj",
        )
        d = adj.to_dict()
        self.assertNotIn("frozen", d)


class TestAdjustmentLogAppendOnly(unittest.TestCase):
    """AdjustmentLog preserves insertion order."""

    def test_empty_log(self):
        log = AdjustmentLog()
        self.assertEqual(log.entries, [])
        self.assertEqual(log.to_list(), [])

    def test_single_entry(self):
        log = AdjustmentLog()
        adj = CriteriaAdjustment(
            criterion_id="s1",
            before={"a": 1},
            after={"a": 2},
            reason="single",
        )
        log.record(adj)
        self.assertEqual(len(log.entries), 1)
        self.assertEqual(log.entries[0].criterion_id, "s1")

    def test_order_preserved(self):
        log = AdjustmentLog()
        for i in range(5):
            log.record(CriteriaAdjustment(
                criterion_id=f"ord_{i}",
                before={"v": i},
                after={"v": i + 1},
                reason=f"step {i}",
            ))
        ids = [e.criterion_id for e in log.entries]
        self.assertEqual(ids, [f"ord_{i}" for i in range(5)])

    def test_entries_returns_copy(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment(
            criterion_id="cp",
            before={"a": 0},
            after={"a": 1},
            reason="copy check",
        ))
        entries = log.entries
        entries.clear()
        self.assertEqual(len(log.entries), 1)

    def test_to_list_matches_entries(self):
        log = AdjustmentLog()
        for i in range(3):
            log.record(CriteriaAdjustment(
                criterion_id=f"tl_{i}",
                before={"x": i},
                after={"x": i + 10},
                reason=f"reason {i}",
            ))
        lst = log.to_list()
        self.assertEqual(len(lst), 3)
        for i, d in enumerate(lst):
            self.assertEqual(d["criterion_id"], f"tl_{i}")


class TestAdjustmentLogSaveLoad(unittest.TestCase):
    """Save and load round-trip."""

    def _make_log(self, n=3):
        log = AdjustmentLog()
        for i in range(n):
            log.record(CriteriaAdjustment(
                criterion_id=f"rt_{i}",
                before={"val": i},
                after={"val": i + 100},
                reason=f"reason_{i}",
            ))
        return log

    def test_round_trip_count(self):
        log = self._make_log(5)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "adj.json"
            log.save(p)
            loaded = AdjustmentLog.load(p)
            self.assertEqual(len(loaded.entries), 5)

    def test_round_trip_ids_preserved(self):
        log = self._make_log(4)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "adj.json"
            log.save(p)
            loaded = AdjustmentLog.load(p)
            orig_ids = [e.criterion_id for e in log.entries]
            loaded_ids = [e.criterion_id for e in loaded.entries]
            self.assertEqual(orig_ids, loaded_ids)

    def test_round_trip_values_preserved(self):
        log = self._make_log(2)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "adj.json"
            log.save(p)
            loaded = AdjustmentLog.load(p)
            for orig, reloaded in zip(log.entries, loaded.entries):
                self.assertEqual(orig.before, reloaded.before)
                self.assertEqual(orig.after, reloaded.after)
                self.assertEqual(orig.reason, reloaded.reason)

    def test_saved_file_valid_json(self):
        log = self._make_log()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "adj.json"
            log.save(p)
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertIn("schema_version", data)
            self.assertIn("adjustments", data)
            self.assertIn("saved_at", data)

    def test_schema_version_in_file(self):
        log = self._make_log(1)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "adj.json"
            log.save(p)
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(data["schema_version"], "criteria_adjustment_log_v1")

    def test_empty_log_round_trip(self):
        log = AdjustmentLog()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "empty.json"
            log.save(p)
            loaded = AdjustmentLog.load(p)
            self.assertEqual(len(loaded.entries), 0)


class TestAdjustmentLogLoadErrors(unittest.TestCase):
    """Load rejects invalid files."""

    def test_wrong_schema_raises(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.json"
            p.write_text(json.dumps({
                "schema_version": "wrong_v99",
                "adjustments": [],
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                AdjustmentLog.load(p)

    def test_missing_schema_raises(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "noschema.json"
            p.write_text(json.dumps({"adjustments": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                AdjustmentLog.load(p)

    def test_missing_adjustments_loads_empty(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "noadj.json"
            p.write_text(json.dumps({
                "schema_version": "criteria_adjustment_log_v1",
            }), encoding="utf-8")
            loaded = AdjustmentLog.load(p)
            self.assertEqual(len(loaded.entries), 0)

    def test_invalid_json_raises(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "garbage.json"
            p.write_text("not json at all", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                AdjustmentLog.load(p)


class TestAdjustmentLogChainedSaves(unittest.TestCase):
    """Multiple save/load cycles maintain integrity."""

    def test_save_load_append_save_load(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "chain.json"
            log1 = AdjustmentLog()
            log1.record(CriteriaAdjustment(
                criterion_id="ch1",
                before={"v": 0},
                after={"v": 1},
                reason="first",
            ))
            log1.save(p)

            log2 = AdjustmentLog.load(p)
            log2.record(CriteriaAdjustment(
                criterion_id="ch2",
                before={"v": 1},
                after={"v": 2},
                reason="second",
            ))
            log2.save(p)

            log3 = AdjustmentLog.load(p)
            self.assertEqual(len(log3.entries), 2)
            self.assertEqual(log3.entries[0].criterion_id, "ch1")
            self.assertEqual(log3.entries[1].criterion_id, "ch2")

    def test_overwrite_preserves_latest(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "overwrite.json"
            for i in range(5):
                log = AdjustmentLog()
                log.record(CriteriaAdjustment(
                    criterion_id=f"ow_{i}",
                    before={"n": i},
                    after={"n": i + 1},
                    reason=f"overwrite {i}",
                ))
                log.save(p)
            loaded = AdjustmentLog.load(p)
            self.assertEqual(len(loaded.entries), 1)
            self.assertEqual(loaded.entries[0].criterion_id, "ow_4")

    def test_nested_directory_creation(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a" / "b" / "c" / "deep.json"
            log = AdjustmentLog()
            log.record(CriteriaAdjustment(
                criterion_id="deep",
                before={"d": 0},
                after={"d": 1},
                reason="deep path",
            ))
            log.save(p)
            self.assertTrue(p.exists())
            loaded = AdjustmentLog.load(p)
            self.assertEqual(len(loaded.entries), 1)


class TestAdjustmentFrozenEnforcement(unittest.TestCase):
    """Frozen flag blocks all adjustments."""

    def test_frozen_true_always_raises(self):
        for criterion_id in ("bench_1", "frozen_test", "immutable"):
            with self.assertRaises(ValueError):
                CriteriaAdjustment(
                    criterion_id=criterion_id,
                    before={"a": 1},
                    after={"a": 2},
                    reason="valid reason",
                    frozen=True,
                )

    def test_frozen_false_allows(self):
        adj = CriteriaAdjustment(
            criterion_id="flex",
            before={"a": 1},
            after={"a": 2},
            reason="allowed",
            frozen=False,
        )
        self.assertEqual(adj.criterion_id, "flex")

    def test_frozen_error_mentions_criterion(self):
        try:
            CriteriaAdjustment(
                criterion_id="my_criterion",
                before={"a": 1},
                after={"a": 2},
                reason="test",
                frozen=True,
            )
            self.fail("Should have raised")
        except ValueError as e:
            self.assertIn("my_criterion", str(e))

    def test_frozen_error_mentions_immutable(self):
        try:
            CriteriaAdjustment(
                criterion_id="x",
                before={"a": 1},
                after={"a": 2},
                reason="test",
                frozen=True,
            )
            self.fail("Should have raised")
        except ValueError as e:
            self.assertIn("immutable", str(e).lower())


if __name__ == "__main__":
    unittest.main()
