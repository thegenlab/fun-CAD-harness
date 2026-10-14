"""Targeted tests for the mandatory four-role runtime topology contract."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import tempfile
import unittest
from pathlib import Path

from simulation_ready_cad_workflow.validation_pipeline.contracts.role_roster import (
    AUTHORITY_SURFACE_BY_ROLE,
    HARNESS_CHANGE_REVIEWER_ROLE,
    HARNESS_OPERATOR_ROLE,
    MANDATORY_ROLES,
    MECHANICAL_LOOP_ROLES,
    PROJECT_RESOURCE_REVIEWER_ROLE,
    WORKER_ROLE,
    RoleRosterError,
    assert_role_topology_ready,
    build_role_registration,
    build_role_roster,
    build_role_runtime,
    load_role_roster,
    load_role_runtime,
    prompt_binding_from_source,
    validate_role_roster,
    validate_role_runtime,
    write_role_roster,
    write_role_runtime,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    canonical_sha256,
)


_REQUIREMENTS_SHA256 = "1" * 64
_STARTING_COMMIT = "2" * 40
_TIMESTAMP = datetime(2026, 8, 14, tzinfo=timezone.utc).isoformat()


class _TopologyFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.project = root / "projects" / "fixture"
        prompt_dir = root / "prompts"
        prompt_dir.mkdir()
        self.registrations = {}
        for index, role_id in enumerate(MANDATORY_ROLES, start=1):
            source = f"prompts/{role_id}.txt"
            (root / source).write_text(
                f"Hard role contract for {role_id}.\n", encoding="utf-8"
            )
            self.registrations[role_id] = build_role_registration(
                provider="codex",
                model="gpt-role-test",
                visibility_surface="desktop_task",
                visible_session_ref=f"visible-session-{index}",
                capability_receipt_sha256=f"{index:x}" * 64,
                prompt_binding=prompt_binding_from_source(root, source),
            )

    def roster(self):
        return build_role_roster(
            workspace_root=self.root,
            project_id="fixture",
            requirements_sha256=_REQUIREMENTS_SHA256,
            starting_commit=_STARTING_COMMIT,
            registrations=self.registrations,
            created_at=_TIMESTAMP,
        )


def _rehash_roster(roster: dict) -> None:
    roster["roster_sha256"] = canonical_sha256(
        {key: value for key, value in roster.items() if key != "roster_sha256"}
    )


def _rehash_runtime(runtime: dict) -> None:
    runtime["runtime_sha256"] = canonical_sha256(
        {key: value for key, value in runtime.items() if key != "runtime_sha256"}
    )


class FourRoleRosterTest(unittest.TestCase):
    def test_roster_optionally_freezes_reasoning_effort(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _TopologyFixture(Path(temporary))
            for registration in fixture.registrations.values():
                registration["reasoning_effort"] = "max"

            roster = fixture.roster()

            self.assertEqual(
                {
                    role["reasoning_effort"]
                    for role in roster["roles"].values()
                },
                {"max"},
            )

    def test_roster_binds_exact_roles_scope_authority_and_visible_sessions(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _TopologyFixture(Path(temporary))
            roster = fixture.roster()

            self.assertEqual(tuple(roster["roles"]), MANDATORY_ROLES)
            self.assertEqual(
                len(
                    {
                        role["visible_session_ref"]
                        for role in roster["roles"].values()
                    }
                ),
                4,
            )
            for role_id, role in roster["roles"].items():
                self.assertEqual(role["role_id"], role_id)
                self.assertEqual(role["project_id"], "fixture")
                self.assertEqual(
                    role["requirements_sha256"], _REQUIREMENTS_SHA256
                )
                self.assertEqual(role["starting_commit"], _STARTING_COMMIT)
                self.assertEqual(
                    role["authority_surface"],
                    AUTHORITY_SURFACE_BY_ROLE[role_id],
                )
                self.assertTrue(role["provider"])
                self.assertTrue(role["visibility_surface"])
                self.assertEqual(len(role["capability_receipt_sha256"]), 64)

            self.assertEqual(
                roster["roster_sha256"],
                canonical_sha256(
                    {
                        key: value
                        for key, value in roster.items()
                        if key != "roster_sha256"
                    }
                ),
            )

    def test_missing_role_and_reused_visible_session_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _TopologyFixture(Path(temporary))
            missing = dict(fixture.registrations)
            del missing[HARNESS_CHANGE_REVIEWER_ROLE]
            with self.assertRaises(RoleRosterError) as caught:
                build_role_roster(
                    workspace_root=fixture.root,
                    project_id="fixture",
                    requirements_sha256=_REQUIREMENTS_SHA256,
                    starting_commit=_STARTING_COMMIT,
                    registrations=missing,
                    created_at=_TIMESTAMP,
                )
            self.assertEqual(caught.exception.code, "ROLE_SET_INVALID")

            duplicate = copy.deepcopy(fixture.registrations)
            duplicate[HARNESS_OPERATOR_ROLE]["visible_session_ref"] = duplicate[
                WORKER_ROLE
            ]["visible_session_ref"]
            with self.assertRaises(RoleRosterError) as caught:
                build_role_roster(
                    workspace_root=fixture.root,
                    project_id="fixture",
                    requirements_sha256=_REQUIREMENTS_SHA256,
                    starting_commit=_STARTING_COMMIT,
                    registrations=duplicate,
                    created_at=_TIMESTAMP,
                )
            self.assertEqual(
                caught.exception.code, "ROLE_VISIBLE_SESSION_REUSED"
            )

    def test_scope_and_authority_cannot_be_rebound_under_a_new_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _TopologyFixture(Path(temporary))
            for field, value in (
                ("project_id", "other-project"),
                ("requirements_sha256", "3" * 64),
                ("starting_commit", "4" * 40),
                ("authority_surface", "worker_loop"),
            ):
                roster = copy.deepcopy(fixture.roster())
                roster["roles"][HARNESS_OPERATOR_ROLE][field] = value
                _rehash_roster(roster)
                with self.subTest(field=field):
                    with self.assertRaises(RoleRosterError) as caught:
                        validate_role_roster(roster, workspace_root=fixture.root)
                    self.assertEqual(caught.exception.code, "ROLE_SCOPE_MISMATCH")

    def test_capability_receipt_and_immutable_hash_are_mandatory(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _TopologyFixture(Path(temporary))
            roster = fixture.roster()
            roster["roles"][WORKER_ROLE]["capability_receipt_sha256"] = "bad"
            _rehash_roster(roster)
            with self.assertRaises(RoleRosterError):
                validate_role_roster(roster, workspace_root=fixture.root)

            roster = fixture.roster()
            roster["created_at"] = (
                datetime(2026, 8, 15, tzinfo=timezone.utc).isoformat()
            )
            with self.assertRaises(RoleRosterError) as caught:
                validate_role_roster(roster, workspace_root=fixture.root)
            self.assertEqual(caught.exception.code, "ROLE_ROSTER_HASH_MISMATCH")


class PromptBindingTest(unittest.TestCase):
    def test_prompt_size_hash_and_source_bytes_are_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _TopologyFixture(Path(temporary))
            roster = fixture.roster()
            source = fixture.root / roster["roles"][WORKER_ROLE][
                "prompt_binding"
            ]["source"]
            source.write_text("changed hard role contract\n", encoding="utf-8")
            with self.assertRaises(RoleRosterError) as caught:
                validate_role_roster(roster, workspace_root=fixture.root)
            self.assertEqual(caught.exception.code, "ROLE_PROMPT_BINDING_CHANGED")

    def test_prompt_source_must_be_regular_workspace_relative_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root.parent / f"{root.name}-outside-prompt.txt"
            outside.write_text("outside\n", encoding="utf-8")
            try:
                with self.assertRaises(RoleRosterError) as caught:
                    prompt_binding_from_source(root, f"../{outside.name}")
                self.assertEqual(caught.exception.code, "ROLE_PROMPT_PATH_INVALID")

                (root / "real.txt").write_text("real\n", encoding="utf-8")
                (root / "linked.txt").symlink_to(root / "real.txt")
                with self.assertRaises(RoleRosterError) as caught:
                    prompt_binding_from_source(root, "linked.txt")
                self.assertEqual(caught.exception.code, "ROLE_PROMPT_PATH_INVALID")
            finally:
                outside.unlink(missing_ok=True)


class RoleRuntimeTest(unittest.TestCase):
    def test_default_is_human_mode_with_ai_reviewer_dormant(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _TopologyFixture(Path(temporary))
            roster = fixture.roster()
            runtime = build_role_runtime(
                roster=roster,
                workspace_root=fixture.root,
                updated_at=_TIMESTAMP,
            )
            self.assertEqual(runtime["resource_review_mode"], "HUMAN")
            self.assertEqual(
                runtime["roles"][PROJECT_RESOURCE_REVIEWER_ROLE]["state"],
                "DORMANT",
            )
            self.assertEqual(
                runtime["roles"][HARNESS_OPERATOR_ROLE]["state"],
                "DORMANT",
            )
            self.assertEqual(
                runtime["roles"][HARNESS_OPERATOR_ROLE]["activity"],
                "awaiting_harness_interrupt",
            )
            self.assertEqual(runtime["roster_sha256"], roster["roster_sha256"])
            self.assertEqual(
                runtime["runtime_sha256"],
                canonical_sha256(
                    {
                        key: value
                        for key, value in runtime.items()
                        if key != "runtime_sha256"
                    }
                ),
            )
            assert_role_topology_ready(
                roster=roster,
                runtime=runtime,
                workspace_root=fixture.root,
            )

    def test_ai_mode_makes_interrupt_routable_without_waking_reviewer(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _TopologyFixture(Path(temporary))
            roster = fixture.roster()
            ai_runtime = build_role_runtime(
                roster=roster,
                workspace_root=fixture.root,
                resource_review_mode="AI",
                updated_at=_TIMESTAMP,
            )
            self.assertEqual(
                ai_runtime["roles"][PROJECT_RESOURCE_REVIEWER_ROLE]["state"],
                "DORMANT",
            )

            invalid = build_role_runtime(
                roster=roster,
                workspace_root=fixture.root,
                updated_at=_TIMESTAMP,
            )
            invalid["roles"][PROJECT_RESOURCE_REVIEWER_ROLE]["state"] = "READY"
            _rehash_runtime(invalid)
            with self.assertRaises(RoleRosterError) as caught:
                validate_role_runtime(
                    invalid, roster=roster, workspace_root=fixture.root
                )
            self.assertEqual(
                caught.exception.code, "RESOURCE_REVIEWER_NOT_DORMANT"
            )

            plugin_runtime = build_role_runtime(
                roster=roster,
                workspace_root=fixture.root,
                updated_at=_TIMESTAMP,
            )
            plugin_runtime["roles"][PROJECT_RESOURCE_REVIEWER_ROLE][
                "state"
            ] = "ACTIVE"
            plugin_runtime["roles"][PROJECT_RESOURCE_REVIEWER_ROLE][
                "activity"
            ] = "plugin_implementation:delivered"
            _rehash_runtime(plugin_runtime)
            validate_role_runtime(
                plugin_runtime, roster=roster, workspace_root=fixture.root
            )

            visual_runtime = build_role_runtime(
                roster=roster,
                workspace_root=fixture.root,
                updated_at=_TIMESTAMP,
            )
            visual_runtime["roles"][PROJECT_RESOURCE_REVIEWER_ROLE][
                "state"
            ] = "ACTIVE"
            visual_runtime["roles"][PROJECT_RESOURCE_REVIEWER_ROLE][
                "activity"
            ] = "static_visual_review:delivered"
            _rehash_runtime(visual_runtime)
            validate_role_runtime(
                visual_runtime, roster=roster, workspace_root=fixture.root
            )

    def test_unknown_stale_and_unreachable_probes_all_block_readiness(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _TopologyFixture(Path(temporary))
            roster = fixture.roster()
            for state, role_id in (
                ("UNKNOWN", WORKER_ROLE),
                ("STALE", HARNESS_OPERATOR_ROLE),
                ("UNREACHABLE", HARNESS_CHANGE_REVIEWER_ROLE),
            ):
                runtime = build_role_runtime(
                    roster=roster,
                    workspace_root=fixture.root,
                    updated_at=_TIMESTAMP,
                )
                runtime["roles"][role_id]["state"] = state
                _rehash_runtime(runtime)
                validate_role_runtime(
                    runtime, roster=roster, workspace_root=fixture.root
                )
                with self.subTest(state=state):
                    with self.assertRaises(RoleRosterError) as caught:
                        assert_role_topology_ready(
                            roster=roster,
                            runtime=runtime,
                            workspace_root=fixture.root,
                        )
                    self.assertEqual(
                        caught.exception.code, "ROLE_TOPOLOGY_UNAVAILABLE"
                    )

    def test_runtime_cannot_change_roster_scope_even_with_a_new_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _TopologyFixture(Path(temporary))
            roster = fixture.roster()
            runtime = build_role_runtime(
                roster=roster,
                workspace_root=fixture.root,
                updated_at=_TIMESTAMP,
            )
            runtime["project_id"] = "other-project"
            _rehash_runtime(runtime)
            with self.assertRaises(RoleRosterError) as caught:
                validate_role_runtime(
                    runtime, roster=roster, workspace_root=fixture.root
                )
            self.assertEqual(caught.exception.code, "ROLE_RUNTIME_SCOPE_MISMATCH")


class RoleRosterStorageTest(unittest.TestCase):
    def test_roster_is_immutable_and_runtime_updates_use_previous_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _TopologyFixture(Path(temporary))
            roster = fixture.roster()
            write_role_roster(
                fixture.project, roster, workspace_root=fixture.root
            )
            write_role_roster(
                fixture.project, roster, workspace_root=fixture.root
            )
            self.assertEqual(
                load_role_roster(
                    fixture.project, workspace_root=fixture.root
                )["roster_sha256"],
                roster["roster_sha256"],
            )

            changed_roster = copy.deepcopy(roster)
            changed_roster["roles"][WORKER_ROLE]["visible_session_ref"] = (
                "replacement-session"
            )
            _rehash_roster(changed_roster)
            with self.assertRaises(RoleRosterError) as caught:
                write_role_roster(
                    fixture.project,
                    changed_roster,
                    workspace_root=fixture.root,
                )
            self.assertEqual(caught.exception.code, "ROLE_ROSTER_IMMUTABLE")

            runtime = build_role_runtime(
                roster=roster,
                workspace_root=fixture.root,
                updated_at=_TIMESTAMP,
            )
            write_role_runtime(
                fixture.project,
                runtime,
                roster=roster,
                workspace_root=fixture.root,
            )
            updated = copy.deepcopy(runtime)
            updated["roles"][WORKER_ROLE]["activity"] = "awaiting_action"
            updated["updated_at"] = datetime(
                2026, 8, 14, 0, 0, 1, tzinfo=timezone.utc
            ).isoformat()
            _rehash_runtime(updated)
            with self.assertRaises(RoleRosterError) as caught:
                write_role_runtime(
                    fixture.project,
                    updated,
                    roster=roster,
                    workspace_root=fixture.root,
                    expected_runtime_sha256="f" * 64,
                )
            self.assertEqual(caught.exception.code, "ROLE_RUNTIME_CONFLICT")

            write_role_runtime(
                fixture.project,
                updated,
                roster=roster,
                workspace_root=fixture.root,
                expected_runtime_sha256=runtime["runtime_sha256"],
            )
            loaded = load_role_runtime(
                fixture.project,
                roster=roster,
                workspace_root=fixture.root,
            )
            self.assertEqual(loaded["runtime_sha256"], updated["runtime_sha256"])


class MechanicalLoopRosterTest(unittest.TestCase):
    def test_worker_and_resource_reviewer_roster_is_legal(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _TopologyFixture(Path(temporary))
            two_roles = {
                role_id: fixture.registrations[role_id]
                for role_id in MECHANICAL_LOOP_ROLES
            }
            roster = build_role_roster(
                workspace_root=fixture.root,
                project_id="fixture",
                requirements_sha256=_REQUIREMENTS_SHA256,
                starting_commit=_STARTING_COMMIT,
                registrations=two_roles,
                created_at=_TIMESTAMP,
            )
            self.assertEqual(tuple(roster["roles"]), MECHANICAL_LOOP_ROLES)
            runtime = build_role_runtime(
                roster=roster,
                workspace_root=fixture.root,
                resource_review_mode="AI",
                updated_at=_TIMESTAMP,
            )
            self.assertEqual(
                tuple(runtime["roles"]), MECHANICAL_LOOP_ROLES
            )
            assert_role_topology_ready(
                roster=roster,
                runtime=runtime,
                workspace_root=fixture.root,
            )

    def test_three_role_set_is_still_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _TopologyFixture(Path(temporary))
            missing = dict(fixture.registrations)
            del missing[HARNESS_OPERATOR_ROLE]
            with self.assertRaises(RoleRosterError) as caught:
                build_role_roster(
                    workspace_root=fixture.root,
                    project_id="fixture",
                    requirements_sha256=_REQUIREMENTS_SHA256,
                    starting_commit=_STARTING_COMMIT,
                    registrations=missing,
                    created_at=_TIMESTAMP,
                )
            self.assertEqual(caught.exception.code, "ROLE_SET_INVALID")


if __name__ == "__main__":
    unittest.main()
