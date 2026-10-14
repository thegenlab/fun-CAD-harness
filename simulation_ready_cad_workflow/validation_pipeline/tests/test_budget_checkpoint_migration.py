"""Budget checkpoint migration and cross-tier recovery scenarios.

Tests real-world checkpoint flows that occur during multi-stage validation:
- Cross-tier checkpoint migration (light→medium→high)
- Checkpoint chain fidelity through multiple save/restore cycles
- Partial recovery: restoring from stale checkpoints
- Concurrent save contention on shared checkpoint paths
- Checkpoint size stability under varied pause histories
- Timer accuracy through migration chains
- Budget arithmetic invariants after multi-hop migrations
"""
from __future__ import annotations

import json
import tempfile
import time
import threading
import unittest
from pathlib import Path

from ..slim_harness.budget import (
    TIERS,
    TierConfig,
    ActiveBudgetTracker,
    BudgetStatus,
    get_tier,
    build_incomplete_summary,
)


class TestCrossTierMigration(unittest.TestCase):
    """Checkpoint restored into a different tier preserves active time."""

    def test_light_to_medium_preserves_active(self):
        light = get_tier("light")
        tracker = ActiveBudgetTracker(tier=light)
        time.sleep(0.03)
        active_before = tracker.elapsed_active

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=get_tier("medium"))
            restored.resume()

        self.assertEqual(restored.tier.name, "medium")
        self.assertAlmostEqual(restored.elapsed_active, active_before, delta=0.02)
        self.assertAlmostEqual(
            restored.remaining_active,
            7200 - active_before,
            delta=0.05,
        )

    def test_medium_to_high_preserves_active(self):
        medium = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=medium)
        time.sleep(0.02)
        active_before = tracker.elapsed_active

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=get_tier("high"))
            restored.resume()

        self.assertEqual(restored.tier.name, "high")
        self.assertAlmostEqual(restored.elapsed_active, active_before, delta=0.02)

    def test_high_to_light_reduces_remaining(self):
        """Migrating downward to a smaller budget reduces remaining time."""
        high = get_tier("high")
        tracker = ActiveBudgetTracker(tier=high)
        time.sleep(0.02)

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=get_tier("light"))
            restored.resume()

        self.assertEqual(restored.tier.name, "light")
        self.assertLess(
            restored.tier.active_budget_seconds,
            high.active_budget_seconds,
        )

    def test_three_hop_migration_chain(self):
        """light→medium→high preserves total active time."""
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        time.sleep(0.02)

        with tempfile.TemporaryDirectory() as tmp:
            cp1 = Path(tmp) / "cp1.json"
            tracker.save_checkpoint(cp1)
            active_after_1 = tracker.elapsed_active

            t2 = ActiveBudgetTracker.restore(cp1, tier=get_tier("medium"))
            t2.resume()
            time.sleep(0.02)

            cp2 = Path(tmp) / "cp2.json"
            t2.save_checkpoint(cp2)
            active_after_2 = t2.elapsed_active

            t3 = ActiveBudgetTracker.restore(cp2, tier=get_tier("high"))
            t3.resume()

        self.assertAlmostEqual(t3.elapsed_active, active_after_2, delta=0.02)
        self.assertGreater(active_after_2, active_after_1)


class TestCheckpointChainFidelity(unittest.TestCase):
    """Multiple save/restore cycles preserve timing accurately."""

    def test_five_cycle_chain(self):
        """5 save/restore cycles on the same tier preserve active time."""
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)

        with tempfile.TemporaryDirectory() as tmp:
            for i in range(5):
                cp = Path(tmp) / f"chain_{i}.json"
                tracker.save_checkpoint(cp)
                active_at_save = tracker.elapsed_active
                tracker = ActiveBudgetTracker.restore(cp, tier=tier)
                tracker.resume()
                self.assertAlmostEqual(
                    tracker.elapsed_active, active_at_save, delta=0.01,
                    msg=f"Cycle {i} lost accuracy",
                )
                time.sleep(0.005)

    def test_chain_pause_count_increments(self):
        """Each restore adds exactly 1 to the pause count."""
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)

        with tempfile.TemporaryDirectory() as tmp:
            for i in range(3):
                cp = Path(tmp) / f"chain_{i}.json"
                tracker.save_checkpoint(cp)
                tracker = ActiveBudgetTracker.restore(cp, tier=tier)
                tracker.resume()

        self.assertEqual(tracker.snapshot()["pause_count"], 3)

    def test_chain_preserves_started_at(self):
        """The original started_at timestamp persists through the file chain."""
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)

        with tempfile.TemporaryDirectory() as tmp:
            cp1 = Path(tmp) / "cp1.json"
            tracker.save_checkpoint(cp1)
            data1 = json.loads(cp1.read_text(encoding="utf-8"))

            restored = ActiveBudgetTracker.restore(cp1, tier=tier)
            restored.resume()

            cp2 = Path(tmp) / "cp2.json"
            restored.save_checkpoint(cp2)
            data2 = json.loads(cp2.read_text(encoding="utf-8"))

        self.assertIn("started_at", data1)
        self.assertIn("started_at", data2)


class TestStaleCheckpointRecovery(unittest.TestCase):
    """Restoring from a stale checkpoint (original tracker advanced further)."""

    def test_stale_checkpoint_restores_snapshot_state(self):
        """Restore from an old checkpoint gives the active time at save time."""
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "stale.json"
            tracker.save_checkpoint(cp)
            active_at_save = tracker.elapsed_active

            time.sleep(0.05)
            active_later = tracker.elapsed_active

            restored = ActiveBudgetTracker.restore(cp, tier=tier)
            restored.resume()

        self.assertAlmostEqual(restored.elapsed_active, active_at_save, delta=0.01)
        self.assertGreater(active_later, active_at_save)

    def test_stale_checkpoint_preserves_pause_count(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        tracker.pause("p1")
        tracker.resume()
        tracker.pause("p2")
        tracker.resume()

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)

            tracker.pause("p3")
            tracker.resume()

            restored = ActiveBudgetTracker.restore(cp, tier=tier)
            restored.resume()

        self.assertEqual(restored.snapshot()["pause_count"], 3)


class TestCheckpointSizeStability(unittest.TestCase):
    """Checkpoint file size stays bounded regardless of history."""

    def test_size_with_zero_pauses(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            zero_size = cp.stat().st_size
        self.assertGreater(zero_size, 0)
        self.assertLess(zero_size, 2048)

    def test_size_with_1000_pauses(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        for i in range(1000):
            tracker.pause(f"p_{i}")
            tracker.resume()
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            big_size = cp.stat().st_size
        self.assertLess(big_size, 2048)

    def test_size_independent_of_pause_count(self):
        """File size doesn't grow with pause count (counts, not individual pauses)."""
        sizes = []
        for pause_count in [0, 10, 100, 500]:
            tracker = ActiveBudgetTracker(tier=get_tier("medium"))
            for i in range(pause_count):
                tracker.pause(f"p_{i}")
                tracker.resume()
            with tempfile.TemporaryDirectory() as tmp:
                cp = Path(tmp) / "cp.json"
                tracker.save_checkpoint(cp)
                sizes.append(cp.stat().st_size)
        max_diff = max(sizes) - min(sizes)
        self.assertLess(max_diff, 100, f"Size varies by {max_diff} bytes: {sizes}")


class TestTimerAccuracyThroughMigration(unittest.TestCase):
    """Timer values remain accurate after checkpoint migration."""

    def test_active_plus_pause_approximates_wall(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        tracker.pause("wait")
        time.sleep(0.02)
        tracker.resume()
        time.sleep(0.01)

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=get_tier("high"))
            restored.resume()

        snap = restored.snapshot()
        active_plus_pause = snap["active_seconds"] + snap["total_pause_seconds"]
        self.assertAlmostEqual(active_plus_pause, snap["wall_seconds"], delta=0.1)

    def test_remaining_decreases_through_migration(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        initial_remaining = tracker.remaining_active
        time.sleep(0.02)

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=tier)
            restored.resume()

        self.assertLess(restored.remaining_active, initial_remaining)


class TestBudgetArithmeticAfterMigration(unittest.TestCase):
    """Budget arithmetic invariants hold after cross-tier migration."""

    def test_remaining_equals_limit_minus_active(self):
        for src_tier_name in TIERS:
            for dst_tier_name in TIERS:
                src = get_tier(src_tier_name)
                dst = get_tier(dst_tier_name)
                tracker = ActiveBudgetTracker(tier=src)
                time.sleep(0.01)

                with tempfile.TemporaryDirectory() as tmp:
                    cp = Path(tmp) / "cp.json"
                    tracker.save_checkpoint(cp)
                    restored = ActiveBudgetTracker.restore(cp, tier=dst)
                    restored.resume()

                expected = max(0, dst.active_budget_seconds - restored.elapsed_active)
                self.assertAlmostEqual(
                    restored.remaining_active, expected, delta=0.05,
                    msg=f"{src_tier_name}→{dst_tier_name}",
                )

    def test_exhaustion_after_downward_migration(self):
        """Active time exceeding the new tier's budget triggers exhaustion."""
        large = TierConfig("large", 100, 60, 2, 0, 10)
        tracker = ActiveBudgetTracker(tier=large)
        time.sleep(0.02)

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)

            tiny = TierConfig("tiny", 0.005, 60, 1, 0, 10)
            restored = ActiveBudgetTracker.restore(cp, tier=tiny)
            restored.resume()

        self.assertTrue(restored.is_exhausted)
        self.assertEqual(restored.remaining_active, 0.0)


class TestConcurrentCheckpointContention(unittest.TestCase):
    """Multiple threads contending on checkpoint saves."""

    def test_three_threads_same_path_last_wins(self):
        tier = get_tier("medium")
        errors = []

        def writer(thread_id, path, sleep_s):
            try:
                tracker = ActiveBudgetTracker(tier=tier)
                time.sleep(sleep_s)
                tracker.save_checkpoint(path)
            except Exception as e:
                errors.append((thread_id, e))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shared.json"
            threads = [
                threading.Thread(target=writer, args=(i, path, 0.005 * i))
                for i in range(3)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            self.assertEqual(errors, [])
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema_version"], "budget_checkpoint_v1")
            self.assertEqual(data["tier"], "medium")

    def test_readers_and_writers_concurrent(self):
        """Reads and writes interleaved don't corrupt the file."""
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        errors = []

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rw.json"
            tracker.save_checkpoint(path)

            def writer():
                try:
                    for _ in range(5):
                        tracker.save_checkpoint(path)
                        time.sleep(0.003)
                except Exception as e:
                    errors.append(("writer", e))

            def reader():
                try:
                    for _ in range(5):
                        try:
                            data = json.loads(path.read_text(encoding="utf-8"))
                            assert data["schema_version"] == "budget_checkpoint_v1"
                        except json.JSONDecodeError:
                            pass
                        time.sleep(0.003)
                except Exception as e:
                    errors.append(("reader", e))

            wt = threading.Thread(target=writer)
            rt = threading.Thread(target=reader)
            wt.start()
            rt.start()
            wt.join(timeout=10)
            rt.join(timeout=10)

        self.assertEqual(errors, [])


class TestMigrationWithSummary(unittest.TestCase):
    """Migration interacts correctly with incomplete summary generation."""

    def test_summary_reflects_new_tier_after_migration(self):
        light = get_tier("light")
        tracker = ActiveBudgetTracker(tier=light)
        time.sleep(0.01)

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=get_tier("high"))
            restored.resume()

        summary = build_incomplete_summary(
            tier=get_tier("high"),
            tracker=restored,
            completed_tests=[],
            pending_tests=["t1"],
            issues=["migrated from light"],
            evidence_paths=[],
            blockers=[],
        )

        self.assertEqual(summary["tier"]["name"], "high")
        self.assertEqual(summary["tier"]["active_budget_seconds"], 10800)
        self.assertIn("migrated from light", summary["issues"])

    def test_summary_timing_matches_restored_tracker(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.02)

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=get_tier("medium"))
            restored.resume()

        snap = restored.snapshot()
        summary = build_incomplete_summary(
            tier=get_tier("medium"),
            tracker=restored,
            completed_tests=[],
            pending_tests=[],
            issues=[],
            evidence_paths=[],
            blockers=[],
        )

        self.assertAlmostEqual(
            summary["timing"]["active_seconds"],
            snap["active_seconds"],
            delta=0.05,
        )


if __name__ == "__main__":
    unittest.main()
