from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path

from simulation_ready_cad_workflow.validation_pipeline.dashboard.cli import (
    main as dashboard_main,
)
from simulation_ready_cad_workflow.validation_pipeline.dashboard.publication import (
    DashboardPublicationError,
    finalize_dashboard_publication,
    load_bound_dashboard_publication,
)
from simulation_ready_cad_workflow.validation_pipeline.dashboard.stage_manifest import (
    STAGE_DECLARATIONS_RELATIVE,
    STAGE_MANIFEST_RELATIVE,
    WorkflowStageManifestError,
    load_workflow_stage_manifest,
    parse_scope_json,
    register_workflow_stage,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    canonical_sha256,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _new_run(
    root: Path,
    *,
    project_id: str = "stage_fixture",
    with_design: bool = True,
) -> Path:
    run = (
        root
        / ".runtime_runs"
        / "projects"
        / project_id
        / "runs"
        / "r01"
    )
    run.mkdir(parents=True)
    if with_design:
        _write_json(run / "inputs" / "design.json", {"design": "fixture"})
    return run


def _evidence(run: Path, name: str = "proof.json") -> Path:
    path = run / "results" / name
    _write_json(path, {"evidence": name})
    return path


def _terminal_binding(publication: dict[str, object]) -> dict[str, object]:
    return {
        "status": "NOT_CERTIFIED",
        "dashboard_publication": {
            "path": publication["path"],
            "sha256": publication["sha256"],
        },
    }


def _recanonicalize_manifest(value: dict[str, object]) -> None:
    value["stages_canonical_sha256"] = canonical_sha256(value["stages"])
    unsigned = {
        key: child
        for key, child in value.items()
        if key != "canonical_payload_sha256"
    }
    value["canonical_payload_sha256"] = canonical_sha256(unsigned)


class DashboardStageManifestTests(unittest.TestCase):
    def test_project_specific_stages_are_dynamic_and_same_kind_is_repeatable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _new_run(root)
            first = _evidence(run, "left.json")
            second = _evidence(run, "right.json")
            register_workflow_stage(
                workspace_root=root,
                run_dir=run,
                stage_id="calibration:left",
                kind="calibration",
                label="左轴标定",
                evidence_paths=[first],
                depends_on=["design-package"],
                scope={"axis_id": "left"},
            )
            register_workflow_stage(
                workspace_root=root,
                run_dir=run,
                stage_id="calibration:right",
                kind="calibration",
                label="右轴标定",
                evidence_paths=[second],
                required=False,
                depends_on=["design-package"],
                scope={"axis_id": "right"},
            )

            publication_ref = finalize_dashboard_publication(
                workspace_root=root,
                run_dir=run,
                terminal_status={"status": "NOT_CERTIFIED"},
            )
            manifest, manifest_ref = load_workflow_stage_manifest(
                workspace_root=root,
                run_dir=run,
            )
            publication, reason = load_bound_dashboard_publication(
                workspace_root=root,
                run_dir=run,
                terminal_status=_terminal_binding(publication_ref),
                require_inventory=False,
            )

            self.assertEqual(reason, "ready")
            self.assertIsNotNone(publication)
            self.assertEqual(manifest["stage_count"], 3)
            self.assertEqual(manifest["automatic_stage_count"], 1)
            self.assertEqual(manifest["custom_stage_count"], 2)
            self.assertEqual(
                [stage["id"] for stage in manifest["stages"]],
                [
                    "design-package",
                    "calibration:left",
                    "calibration:right",
                ],
            )
            self.assertEqual(
                [stage["kind"] for stage in manifest["stages"][1:]],
                ["calibration", "calibration"],
            )
            self.assertEqual(
                [stage["status"] for stage in manifest["stages"][1:]],
                ["EVIDENCE_COMPLETE", "EVIDENCE_COMPLETE"],
            )
            self.assertFalse(manifest["stages"][2]["required"])
            self.assertEqual(
                publication["workflow_stage_manifest"],
                manifest_ref,
            )
            self.assertEqual(publication["stages"], manifest["stages"])

    def test_no_declaration_still_materializes_full_dynamic_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _new_run(root)
            _write_json(
                run / "plans" / "input.json",
                {"capsule_id": "input"},
            )
            _write_json(
                run / "plans" / "output.json",
                {"capsule_id": "output"},
            )

            publication_ref = finalize_dashboard_publication(
                workspace_root=root,
                run_dir=run,
                terminal_status={"status": "NOT_CERTIFIED"},
            )
            manifest, _binding = load_workflow_stage_manifest(
                workspace_root=root,
                run_dir=run,
            )

            self.assertEqual(manifest["declaration_count"], 0)
            self.assertEqual(manifest["stage_count"], 3)
            self.assertEqual(
                [stage["id"] for stage in manifest["stages"]],
                [
                    "design-package",
                    "mechanism-plan:input",
                    "mechanism-plan:output",
                ],
            )
            self.assertEqual(publication_ref["stage_count"], 3)
            self.assertTrue((run / STAGE_MANIFEST_RELATIVE).is_file())

    def test_cli_is_the_standard_registration_route_and_has_no_status(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _new_run(root)
            proof = _evidence(run)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = dashboard_main(
                    [
                        "register-stage",
                        "--workspace-root",
                        str(root),
                        "--run",
                        _relative(root, run),
                        "--id",
                        "inspection:drive",
                        "--kind",
                        "inspection",
                        "--label",
                        "驱动检查",
                        "--depends-on",
                        "design-package",
                        "--scope-json",
                        '{"part":"drive"}',
                        "--evidence",
                        _relative(root, proof),
                    ]
                )
            self.assertEqual(result, 0)
            self.assertEqual(
                json.loads(output.getvalue())["stage_id"],
                "inspection:drive",
            )
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    dashboard_main(
                        [
                            "register-stage",
                            "--workspace-root",
                            str(root),
                            "--run",
                            _relative(root, run),
                            "--id",
                            "fake-pass",
                            "--kind",
                            "inspection",
                            "--label",
                            "Fake",
                            "--evidence",
                            _relative(root, proof),
                            "--status",
                            "PASSED",
                        ]
                    )

    def test_generic_terminal_labels_and_declaration_status_are_rejected(
        self,
    ) -> None:
        for label in ("PASS", "FAILED", "PASSED", "RESOLVED"):
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    run = _new_run(root)
                    with self.assertRaises(DashboardPublicationError) as raised:
                        finalize_dashboard_publication(
                            workspace_root=root,
                            run_dir=run,
                            terminal_status={"status": label},
                        )
                    self.assertEqual(
                        raised.exception.code,
                        "DASHBOARD_PUBLICATION_TERMINAL_STATUS_INVALID",
                    )

        for label in ("PASS", "FAIL", "RESOLVED"):
            with self.subTest(declaration_status=label):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    run = _new_run(root)
                    registration = register_workflow_stage(
                        workspace_root=root,
                        run_dir=run,
                        stage_id="review:one",
                        kind="review",
                        label="Review",
                        evidence_paths=[_evidence(run)],
                    )
                    declaration_path = root / str(registration["path"])
                    declaration = json.loads(
                        declaration_path.read_text(encoding="utf-8")
                    )
                    declaration["status"] = label
                    _write_json(declaration_path, declaration)
                    with self.assertRaises(
                        DashboardPublicationError
                    ) as raised:
                        finalize_dashboard_publication(
                            workspace_root=root,
                            run_dir=run,
                            terminal_status={"status": "NOT_CERTIFIED"},
                        )
                    self.assertEqual(
                        raised.exception.code,
                        "DASHBOARD_STAGE_DECLARATION_FIELDS_INVALID",
                    )

    def test_registration_rejects_traversal_symlink_drift_and_missing_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _new_run(root)
            outside = root / "outside.json"
            _write_json(outside, {"outside": True})
            with self.assertRaises(WorkflowStageManifestError) as raised:
                register_workflow_stage(
                    workspace_root=root,
                    run_dir=run,
                    stage_id="outside",
                    kind="review",
                    label="Outside",
                    evidence_paths=[outside],
                )
            self.assertEqual(
                raised.exception.code,
                "DASHBOARD_STAGE_EVIDENCE_OUTSIDE_RUN",
            )

            target = _evidence(run, "target.json")
            link = run / "results" / "link.json"
            link.symlink_to(target)
            with self.assertRaises(WorkflowStageManifestError) as raised:
                register_workflow_stage(
                    workspace_root=root,
                    run_dir=run,
                    stage_id="symlink",
                    kind="review",
                    label="Symlink",
                    evidence_paths=[link],
                )
            self.assertEqual(
                raised.exception.code,
                "DASHBOARD_STAGE_EVIDENCE_SYMLINK",
            )

            drift = _evidence(run, "drift.json")
            register_workflow_stage(
                workspace_root=root,
                run_dir=run,
                stage_id="drift",
                kind="review",
                label="Drift",
                evidence_paths=[drift],
            )
            _write_json(drift, {"changed": True})
            with self.assertRaises(DashboardPublicationError) as raised:
                finalize_dashboard_publication(
                    workspace_root=root,
                    run_dir=run,
                    terminal_status={"status": "NOT_CERTIFIED"},
                )
            self.assertEqual(
                raised.exception.code,
                "DASHBOARD_STAGE_EVIDENCE_HASH_MISMATCH",
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _new_run(root)
            proof = _evidence(run)
            register_workflow_stage(
                workspace_root=root,
                run_dir=run,
                stage_id="missing",
                kind="review",
                label="Missing",
                evidence_paths=[proof],
            )
            proof.unlink()
            with self.assertRaises(DashboardPublicationError) as raised:
                finalize_dashboard_publication(
                    workspace_root=root,
                    run_dir=run,
                    terminal_status={"status": "NOT_CERTIFIED"},
                )
            self.assertEqual(
                raised.exception.code,
                "DASHBOARD_STAGE_EVIDENCE_MISSING",
            )

    def test_replay_concurrency_and_terminal_mutation_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _new_run(root)
            proof = _evidence(run)
            register_workflow_stage(
                workspace_root=root,
                run_dir=run,
                stage_id="unique",
                kind="review",
                label="Unique",
                evidence_paths=[proof],
            )
            with self.assertRaises(WorkflowStageManifestError) as raised:
                register_workflow_stage(
                    workspace_root=root,
                    run_dir=run,
                    stage_id="unique",
                    kind="review",
                    label="Unique",
                    evidence_paths=[proof],
                )
            self.assertEqual(
                raised.exception.code,
                "DASHBOARD_STAGE_REGISTRATION_REPLAY",
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _new_run(root)
            proof = _evidence(run)
            barrier = threading.Barrier(2)
            outcomes: list[str] = []

            def register() -> None:
                barrier.wait()
                try:
                    register_workflow_stage(
                        workspace_root=root,
                        run_dir=run,
                        stage_id="racing",
                        kind="review",
                        label="Racing",
                        evidence_paths=[proof],
                    )
                except WorkflowStageManifestError as exc:
                    outcomes.append(exc.code)
                else:
                    outcomes.append("ok")

            threads = [
                threading.Thread(target=register),
                threading.Thread(target=register),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertCountEqual(
                outcomes,
                ["ok", "DASHBOARD_STAGE_REGISTRATION_REPLAY"],
            )

            _write_json(
                run / "run_status.json",
                {"status": "NOT_CERTIFIED"},
            )
            with self.assertRaises(WorkflowStageManifestError) as raised:
                register_workflow_stage(
                    workspace_root=root,
                    run_dir=run,
                    stage_id="after-terminal",
                    kind="review",
                    label="After terminal",
                    evidence_paths=[proof],
                )
            self.assertEqual(
                raised.exception.code,
                "DASHBOARD_STAGE_RUN_TERMINAL",
            )

    def test_dependency_missing_cycle_and_extra_registry_entry_block_publish(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _new_run(root)
            register_workflow_stage(
                workspace_root=root,
                run_dir=run,
                stage_id="design-package",
                kind="override",
                label="Attempted override",
                evidence_paths=[_evidence(run)],
            )
            with self.assertRaises(DashboardPublicationError) as raised:
                finalize_dashboard_publication(
                    workspace_root=root,
                    run_dir=run,
                    terminal_status={"status": "NOT_CERTIFIED"},
                )
            self.assertEqual(
                raised.exception.code,
                "DASHBOARD_STAGE_ID_COLLISION",
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _new_run(root)
            register_workflow_stage(
                workspace_root=root,
                run_dir=run,
                stage_id="missing-dependency",
                kind="review",
                label="Missing dependency",
                evidence_paths=[_evidence(run)],
                depends_on=["does-not-exist"],
            )
            with self.assertRaises(DashboardPublicationError) as raised:
                finalize_dashboard_publication(
                    workspace_root=root,
                    run_dir=run,
                    terminal_status={"status": "NOT_CERTIFIED"},
                )
            self.assertEqual(
                raised.exception.code,
                "DASHBOARD_STAGE_DEPENDENCY_MISSING",
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _new_run(root)
            proof = _evidence(run)
            register_workflow_stage(
                workspace_root=root,
                run_dir=run,
                stage_id="cycle:a",
                kind="review",
                label="Cycle A",
                evidence_paths=[proof],
                depends_on=["cycle:b"],
            )
            register_workflow_stage(
                workspace_root=root,
                run_dir=run,
                stage_id="cycle:b",
                kind="review",
                label="Cycle B",
                evidence_paths=[proof],
                depends_on=["cycle:a"],
            )
            with self.assertRaises(DashboardPublicationError) as raised:
                finalize_dashboard_publication(
                    workspace_root=root,
                    run_dir=run,
                    terminal_status={"status": "NOT_CERTIFIED"},
                )
            self.assertEqual(
                raised.exception.code,
                "DASHBOARD_STAGE_DEPENDENCY_CYCLE",
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _new_run(root)
            register_workflow_stage(
                workspace_root=root,
                run_dir=run,
                stage_id="valid",
                kind="review",
                label="Valid",
                evidence_paths=[_evidence(run)],
            )
            extra = run / STAGE_DECLARATIONS_RELATIVE / "untracked.txt"
            extra.write_text("not a declaration", encoding="utf-8")
            with self.assertRaises(DashboardPublicationError) as raised:
                finalize_dashboard_publication(
                    workspace_root=root,
                    run_dir=run,
                    terminal_status={"status": "NOT_CERTIFIED"},
                )
            self.assertEqual(
                raised.exception.code,
                "DASHBOARD_STAGE_DECLARATION_EXTRA_ENTRY",
            )

    def test_strict_json_rejects_duplicate_and_all_nonfinite_numbers(
        self,
    ) -> None:
        attacks = {
            "duplicate": (
                '{"schema_version":"a","schema_version":"b"}\n',
                "DASHBOARD_STAGE_JSON_DUPLICATE_KEY",
            ),
            "nan": ('{"value":NaN}\n', "DASHBOARD_STAGE_JSON_NONFINITE"),
            "infinity": (
                '{"value":Infinity}\n',
                "DASHBOARD_STAGE_JSON_NONFINITE",
            ),
            "overflow": (
                '{"value":1e999}\n',
                "DASHBOARD_STAGE_JSON_NONFINITE",
            ),
        }
        for name, (payload, expected_code) in attacks.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    run = _new_run(root)
                    registration = register_workflow_stage(
                        workspace_root=root,
                        run_dir=run,
                        stage_id="strict",
                        kind="review",
                        label="Strict",
                        evidence_paths=[_evidence(run)],
                    )
                    declaration_path = root / str(registration["path"])
                    declaration_path.write_text(payload, encoding="utf-8")
                    with self.assertRaises(
                        DashboardPublicationError
                    ) as raised:
                        finalize_dashboard_publication(
                            workspace_root=root,
                            run_dir=run,
                            terminal_status={"status": "NOT_CERTIFIED"},
                        )
                    self.assertEqual(raised.exception.code, expected_code)

        for payload in (
            '{"value":1,"value":2}',
            '{"value":NaN}',
            '{"value":Infinity}',
            '{"value":1e999}',
        ):
            with self.subTest(scope=payload):
                with self.assertRaises(WorkflowStageManifestError):
                    parse_scope_json(payload)

    def test_handcrafted_publication_and_manifest_mismatch_are_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _new_run(root)
            publication_ref = finalize_dashboard_publication(
                workspace_root=root,
                run_dir=run,
                terminal_status={"status": "NOT_CERTIFIED"},
            )
            publication_path = root / str(publication_ref["path"])
            publication = json.loads(
                publication_path.read_text(encoding="utf-8")
            )
            publication["stages"][0]["label"] = "handcrafted publication"
            publication["stage_manifest_sha256"] = canonical_sha256(
                publication["stages"]
            )
            _write_json(publication_path, publication)
            status = {
                "status": "NOT_CERTIFIED",
                "dashboard_publication": {
                    "path": _relative(root, publication_path),
                    "sha256": _sha256(publication_path),
                },
            }
            loaded, reason = load_bound_dashboard_publication(
                workspace_root=root,
                run_dir=run,
                terminal_status=status,
                require_inventory=False,
            )
            self.assertIsNone(loaded)
            self.assertEqual(
                reason,
                "dashboard-stage-manifest-binding-invalid",
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _new_run(root)
            publication_ref = finalize_dashboard_publication(
                workspace_root=root,
                run_dir=run,
                terminal_status={"status": "NOT_CERTIFIED"},
            )
            publication_path = root / str(publication_ref["path"])
            manifest_path = run / STAGE_MANIFEST_RELATIVE
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            manifest["stages"][0]["label"] = "handcrafted manifest"
            _recanonicalize_manifest(manifest)
            _write_json(manifest_path, manifest)
            publication = json.loads(
                publication_path.read_text(encoding="utf-8")
            )
            publication["workflow_stage_manifest"] = {
                "path": _relative(root, manifest_path),
                "sha256": _sha256(manifest_path),
                "size_bytes": manifest_path.stat().st_size,
                "canonical_payload_sha256": manifest[
                    "canonical_payload_sha256"
                ],
            }
            _write_json(publication_path, publication)
            status = {
                "status": "NOT_CERTIFIED",
                "dashboard_publication": {
                    "path": _relative(root, publication_path),
                    "sha256": _sha256(publication_path),
                },
            }
            loaded, reason = load_bound_dashboard_publication(
                workspace_root=root,
                run_dir=run,
                terminal_status=status,
                require_inventory=False,
            )
            self.assertIsNone(loaded)
            self.assertEqual(
                reason,
                "dashboard-stage-manifest-binding-invalid",
            )

    def test_traversal_and_symlink_manifest_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _new_run(root)
            registration = register_workflow_stage(
                workspace_root=root,
                run_dir=run,
                stage_id="normalized",
                kind="review",
                label="Normalized",
                evidence_paths=[_evidence(run)],
            )
            declaration_path = root / str(registration["path"])
            declaration = json.loads(
                declaration_path.read_text(encoding="utf-8")
            )
            declaration["evidence"][0]["path"] = (
                _relative(root, run)
                + "/results/../results/proof.json"
            )
            unsigned = {
                key: child
                for key, child in declaration.items()
                if key != "declaration_payload_sha256"
            }
            declaration["declaration_payload_sha256"] = canonical_sha256(
                unsigned
            )
            _write_json(declaration_path, declaration)
            with self.assertRaises(DashboardPublicationError) as raised:
                finalize_dashboard_publication(
                    workspace_root=root,
                    run_dir=run,
                    terminal_status={"status": "NOT_CERTIFIED"},
                )
            self.assertEqual(
                raised.exception.code,
                "DASHBOARD_STAGE_EVIDENCE_PATH_INVALID",
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _new_run(root)
            outside = root / "manifest.json"
            _write_json(outside, {"fake": True})
            manifest_path = run / STAGE_MANIFEST_RELATIVE
            manifest_path.parent.mkdir(parents=True)
            manifest_path.symlink_to(outside)
            with self.assertRaises(DashboardPublicationError) as raised:
                finalize_dashboard_publication(
                    workspace_root=root,
                    run_dir=run,
                    terminal_status={"status": "NOT_CERTIFIED"},
                )
            self.assertEqual(
                raised.exception.code,
                "DASHBOARD_STAGE_MANIFEST_JSON_INVALID",
            )

    def test_inventory_must_cover_publication_manifest_and_declarations(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _new_run(root)
            register_workflow_stage(
                workspace_root=root,
                run_dir=run,
                stage_id="inventory",
                kind="review",
                label="Inventory",
                evidence_paths=[_evidence(run)],
            )
            publication_ref = finalize_dashboard_publication(
                workspace_root=root,
                run_dir=run,
                terminal_status={"status": "NOT_CERTIFIED"},
            )
            inventory_path = run / "artifacts.sha256.json"
            artifacts = {
                _relative(root, path): _sha256(path)
                for path in run.rglob("*")
                if path.is_file()
                and not path.is_symlink()
                and path != inventory_path
            }
            inventory = {
                "schema_version": "simulation_ready_cad_artifact_index_v1",
                "path_base": "workspace_root",
                "artifacts": artifacts,
            }
            _write_json(inventory_path, inventory)
            status = {
                **_terminal_binding(publication_ref),
                "artifact_inventory": _relative(root, inventory_path),
                "artifact_inventory_sha256": _sha256(inventory_path),
            }
            loaded, reason = load_bound_dashboard_publication(
                workspace_root=root,
                run_dir=run,
                terminal_status=status,
                require_inventory=True,
            )
            self.assertEqual(reason, "ready")
            self.assertIsNotNone(loaded)

            manifest_path = run / STAGE_MANIFEST_RELATIVE
            inventory["artifacts"].pop(_relative(root, manifest_path))
            _write_json(inventory_path, inventory)
            status["artifact_inventory_sha256"] = _sha256(inventory_path)
            loaded, reason = load_bound_dashboard_publication(
                workspace_root=root,
                run_dir=run,
                terminal_status=status,
                require_inventory=True,
            )
            self.assertIsNone(loaded)
            self.assertEqual(reason, "terminal-inventory-schema-invalid")

    def test_materialization_seals_registry_and_rebuilds_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _new_run(root)
            proof = _evidence(run)
            publication_ref = finalize_dashboard_publication(
                workspace_root=root,
                run_dir=run,
                terminal_status={"status": "NOT_CERTIFIED"},
            )
            with self.assertRaises(WorkflowStageManifestError) as raised:
                register_workflow_stage(
                    workspace_root=root,
                    run_dir=run,
                    stage_id="too-late",
                    kind="review",
                    label="Too late",
                    evidence_paths=[proof],
                )
            self.assertEqual(
                raised.exception.code,
                "DASHBOARD_STAGE_REGISTRY_SEALED",
            )

            manifest_path = run / STAGE_MANIFEST_RELATIVE
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            manifest["stages"][0]["label"] = "forged"
            _recanonicalize_manifest(manifest)
            _write_json(manifest_path, manifest)
            with self.assertRaises(DashboardPublicationError) as raised:
                finalize_dashboard_publication(
                    workspace_root=root,
                    run_dir=run,
                    terminal_status={"status": "NOT_CERTIFIED"},
                )
            self.assertEqual(
                raised.exception.code,
                "DASHBOARD_STAGE_MANIFEST_REBUILD_MISMATCH",
            )
            self.assertTrue(publication_ref["path"])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _new_run(root)
            finalize_dashboard_publication(
                workspace_root=root,
                run_dir=run,
                terminal_status={"status": "NOT_CERTIFIED"},
            )
            manifest_path = run / STAGE_MANIFEST_RELATIVE
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            duplicate = dict(manifest["stages"][0])
            duplicate["order"] = 2
            manifest["stages"].append(duplicate)
            manifest["automatic_stage_count"] = 2
            manifest["stage_count"] = 2
            _recanonicalize_manifest(manifest)
            _write_json(manifest_path, manifest)
            with self.assertRaises(DashboardPublicationError) as raised:
                finalize_dashboard_publication(
                    workspace_root=root,
                    run_dir=run,
                    terminal_status={"status": "NOT_CERTIFIED"},
                )
            self.assertEqual(
                raised.exception.code,
                "DASHBOARD_STAGE_ID_DUPLICATE",
            )


if __name__ == "__main__":
    unittest.main()
