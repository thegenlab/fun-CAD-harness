"""Cross-revision artifact reuse: content fingerprints, mesh cache, receipts.

Scope and safety rules (deliberate, do not widen casually):

* **Mesh cache** — the solid-to-STL tessellation in family collision
  materialization is a pure function of the solid geometry and the fixed
  mesher parameters.  Cached bytes are hash-verified on fetch and flow
  through every downstream fidelity/audit check unchanged, so a cache hit
  can never weaken validation; a corrupt cache entry degrades to a miss.
* **Lane fingerprints** — every physical lane job gets a content
  fingerprint over its run-identity-free inputs.  Traces and lane results
  embed the exact ``plan_sha256`` / ``load_case_sha256`` of the run that
  *executed* them and the anti-tamper chain in ``finalize.py`` verifies
  those bindings.
* **Sign-off receipts** — a new run may formally sign off a sibling run's
  already-verified physical lane instead of recomputing it.  The receipt
  is the *load-bearing* license: verification keeps rejecting foreign
  trace identities unless a hash-chained receipt in the consuming run
  binds (a) the recomputed content fingerprint of the current inputs,
  (b) the origin run and its plan/load identities, and (c) the byte
  hashes of every adopted artifact.  Deleting or editing a receipt breaks
  the chain and the run fails closed; deleting the whole receipt file
  restores the plain ``TRACE_IDENTITY_MISMATCH`` rejection.  The origin
  run is only ever read, never modified.
* The reuse DAG across revisions is derived read-only from the recorded
  manifests (see ``tools/reuse_dag.py``).
"""

from __future__ import annotations

import datetime as _datetime
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .io import (
    canonical_sha256,
    read_json,
    read_trace_json,
    sha256_file,
    write_json_atomic,
)
from .models import MechanismValidationError

REUSE_MANIFEST_SCHEMA = "mechanism_reuse_manifest_v1"
MESH_CACHE_SCHEMA = "mechanism_mesh_cache_v1"
LANE_FINGERPRINT_SCHEMA = "mechanism_lane_fingerprint_v3"
REUSE_RECEIPT_SCHEMA = "mechanism_reuse_receipt_v1"
REUSE_RECEIPT_CHAIN_SCHEMA = "mechanism_reuse_receipt_chain_v1"

# Artifact labels adopted by a lane sign-off.  The CAD/FCL trace audit is
# deliberately absent: the consuming run always recomputes it itself.
LANE_SIGN_OFF_LABELS = (
    "physical_trace",
    "ideal_reference_trace",
    "ideal_reference_comparison",
    "comparison",
)
LANE_SIGN_OFF_IDEAL_DYNAMICS_LABELS = (
    "ideal_dynamics_trace",
    "ideal_dynamics_comparison",
)

# Plan/document keys that must not enter a cross-run fingerprint. They are
# either run identity (paths, per-run intake evidence), evidence pointers
# (report paths re-written every run), or byte-unstable duplicates of
# content identities that stay in the fingerprint (a mesh path is dropped,
# its sibling content sha256 is kept; the assembly byte hash is dropped,
# per-solid source/component hashes are kept).
_IDENTITY_DROPPED_KEYS = frozenset(
    {
        "package_dir",
        "intake_report",
        "intake_report_sha256",
        "capsule_file",
        "mesh",
        "report",
        "report_sha256",
        "assembly_contract",
        "assembly_file",
        "assembly_sha256",
        "static_geometry_preflight",
        "mesh_cache",
        # Whole-package provenance and per-run integrity-guard evidence.
        # None of these are physics inputs of a lane: the geometry a lane
        # simulates is bound precisely by the per-component mesh sha256s in
        # ``collision_geometry`` (kept), and the behavioural parameters by
        # ``family_spec`` and the load case (kept).  The package binding and
        # per-file artifact hashes are coarser supersets that also cover
        # files a lane never consumes (generator scripts, task mappings),
        # so keeping them would recompute physics for provably irrelevant
        # edits.  Collider fidelity is a gate the *consuming* run re-earns
        # itself at dispatch; a failing gate blocks dispatch before any
        # sign-off can happen.
        "package_binding_sha256",
        "artifact_hashes",
        "package_id",
        "design_version",
        "requirements_sha256",
        "collider_fidelity",
        "home_interpenetration_audit",
    }
)


def _normalized(node: Any, dropped: frozenset[str]) -> Any:
    if isinstance(node, Mapping):
        return {
            key: _normalized(value, dropped)
            for key, value in sorted(node.items())
            if key not in dropped
        }
    if isinstance(node, (list, tuple)):
        return [_normalized(item, dropped) for item in node]
    return node


def lane_fingerprint(
    *,
    plan_document: Mapping[str, Any],
    load_case_document: Mapping[str, Any],
    validator_source_sha256: str,
    plugin_id: str,
    branches: tuple[str, ...],
    task_plugin_binding: Mapping[str, str] | None = None,
    extra_dropped_keys: tuple[str, ...] = (),
) -> str:
    """Content fingerprint of one physical lane job.

    Two lane jobs with equal fingerprints would compute identical traces:
    the fingerprint covers everything the lane execution path consumes —
    the mechanism plan normalized to be run-identity and provenance free,
    the load case, the mechanism plugin, any Reviewer TaskAdapter that owns
    the program, and the validator source hash.

    ``extra_dropped_keys`` lets a family plugin exclude plan sections that
    its own lane execution path provably never consumes (for example the
    acceptance contract, which most families only evaluate in finalize).
    The keys and their justification live on the plugin, next to the code
    that would have to start consuming them.
    """

    dropped = _IDENTITY_DROPPED_KEYS | frozenset(extra_dropped_keys)
    identity = {
        "schema_version": LANE_FINGERPRINT_SCHEMA,
        "validator_source_sha256": validator_source_sha256,
        "plugin_id": plugin_id,
        "branches": sorted(branches),
        "extra_dropped_keys": sorted(extra_dropped_keys),
        "plan": _normalized(dict(plan_document), dropped),
        "load_case": _normalized(dict(load_case_document), dropped),
    }
    adapter_spec = plan_document.get("adapter_spec")
    family_spec = (
        adapter_spec.get("family_spec")
        if isinstance(adapter_spec, Mapping)
        else None
    )
    project_adapter_file = (
        family_spec.get("project_adapter_file")
        if isinstance(family_spec, Mapping)
        else None
    )
    artifact_hashes = plan_document.get("artifact_hashes")
    if isinstance(project_adapter_file, str) and isinstance(
        artifact_hashes, Mapping
    ):
        # ``artifact_hashes`` is otherwise intentionally dropped because it
        # contains many files no lane consumes.  A project MechanismAdapter is
        # different: its Python bytes execute the lane and its digest exists
        # only in this map, so retain that one exact binding.
        identity["project_adapter_source_binding"] = {
            "path": project_adapter_file,
            "sha256": str(artifact_hashes.get(project_adapter_file) or ""),
        }
    if task_plugin_binding is not None:
        identity["task_plugin_binding"] = {
            "plugin_id": str(task_plugin_binding["plugin_id"]),
            "sha256": str(task_plugin_binding["sha256"]),
        }
    return canonical_sha256(identity)


def write_reuse_manifest(
    destination: Path,
    *,
    entries: list[dict[str, Any]],
    mesh_cache_stats: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": REUSE_MANIFEST_SCHEMA,
        "note": (
            "EXECUTED lanes ran their physics in this run. REUSED lanes "
            "adopted a sibling run's verified outputs through a "
            "hash-chained sign-off receipt (results/reuse_receipts.json); "
            "finalize verification refuses adopted evidence without a "
            "valid receipt. Mesh cache entries record byte-level "
            "tessellation reuse."
        ),
        "lane_jobs": entries,
        "mesh_cache": dict(mesh_cache_stats or {}),
    }
    write_json_atomic(destination, manifest)
    return manifest


def reuse_summary_for_run(run_dir: str | Path) -> dict[str, Any] | None:
    """Worker-facing summary of lane-level cross-run reuse for one run.

    Built read-only from the dispatch-owned reuse manifest and plans; the
    summary is observability for the Worker (what was preserved, what was
    recomputed, which planned tests reran their physics), never evidence —
    finalize keeps verifying receipts and artifact hashes independently.
    Acceptance evaluation and certificates always re-derive in finalize
    regardless of lane reuse.
    """

    run = Path(run_dir)
    try:
        manifest = read_json(run / "results" / "reuse_manifest.json")
    except (OSError, ValueError):
        return None
    lane_jobs = manifest.get("lane_jobs")
    if not isinstance(lane_jobs, list):
        return None
    preserved: list[dict[str, Any]] = []
    recomputed: list[dict[str, Any]] = []
    actions: dict[tuple[str, str], str] = {}
    for entry in lane_jobs:
        if not isinstance(entry, dict):
            continue
        lane = {
            "capsule_id": entry.get("capsule_id"),
            "load_case_id": entry.get("load_case_id"),
        }
        actions[
            (str(entry.get("capsule_id")), str(entry.get("load_case_id")))
        ] = str(entry.get("action"))
        if entry.get("action") == "REUSED":
            preserved.append(
                {
                    **lane,
                    "origin_run": entry.get("origin_run"),
                    "receipt_sha256": entry.get("receipt_sha256"),
                }
            )
        else:
            recomputed.append(lane)
    tests: list[dict[str, Any]] = []
    try:
        for plan_path in sorted((run / "plans").glob("*.json")):
            plan = read_json(plan_path)
            capsule_id = str(plan.get("capsule_id") or "")
            contract = plan.get("acceptance_contract")
            requirements = (
                contract.get("requirements")
                if isinstance(contract, dict)
                else None
            )
            for requirement in requirements or []:
                if not isinstance(requirement, dict):
                    continue
                load_case_id = str(
                    requirement.get("test_load_case_id") or ""
                )
                tests.append(
                    {
                        "requirement_id": requirement.get("requirement_id"),
                        "task_id": requirement.get("task_id"),
                        "capsule_id": capsule_id,
                        "load_case_id": load_case_id,
                        "physics": actions.get(
                            (capsule_id, load_case_id), "NOT_DISPATCHED"
                        ),
                    }
                )
    except (OSError, ValueError):
        tests = []
    return {
        "preserved_lane_count": len(preserved),
        "recomputed_lane_count": len(recomputed),
        "preserved": preserved,
        "recomputed": recomputed,
        "tests": tests,
    }


def _receipts_path(output_dir: Path) -> Path:
    return Path(output_dir) / "results" / "reuse_receipts.json"


def _entry_digest(entry: Mapping[str, Any]) -> str:
    unsigned = {
        key: value for key, value in entry.items() if key != "entry_sha256"
    }
    return canonical_sha256(unsigned)


def load_reuse_receipts(output_dir: str | Path) -> list[dict[str, Any]]:
    """Load and verify the run's sign-off receipt chain.

    Returns ``[]`` when the run has no receipts.  Any structural damage —
    unreadable file, unknown schema, broken hash chain, edited entry —
    raises ``REUSE_RECEIPT_CHAIN_INVALID``: a run whose reuse licenses
    cannot be trusted must fail, not silently degrade.
    """

    path = _receipts_path(Path(output_dir))
    if not path.is_file():
        return []
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MechanismValidationError(
            "REUSE_RECEIPT_CHAIN_INVALID",
            f"reuse receipt chain is unreadable: {exc}",
        ) from exc
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != REUSE_RECEIPT_CHAIN_SCHEMA
        or not isinstance(document.get("entries"), list)
    ):
        raise MechanismValidationError(
            "REUSE_RECEIPT_CHAIN_INVALID",
            "reuse receipt chain document has an unknown structure",
        )
    entries: list[dict[str, Any]] = []
    previous: str | None = None
    for index, entry in enumerate(document["entries"]):
        if (
            not isinstance(entry, dict)
            or entry.get("schema_version") != REUSE_RECEIPT_SCHEMA
            or entry.get("previous_entry_sha256") != previous
            or entry.get("entry_sha256") != _entry_digest(entry)
        ):
            raise MechanismValidationError(
                "REUSE_RECEIPT_CHAIN_INVALID",
                f"reuse receipt chain is broken at entry {index}",
            )
        previous = str(entry["entry_sha256"])
        entries.append(entry)
    return entries


def reset_reuse_receipts(output_dir: str | Path) -> None:
    """Start a fresh receipt set for one dispatch attempt.

    Resource retries rebuild the dispatch inside the same run.  Receipts from
    an earlier attempt cannot survive into a later attempt whose promotion
    hint may now be a miss and whose lane will execute normally.
    """

    write_json_atomic(
        _receipts_path(Path(output_dir)),
        {
            "schema_version": REUSE_RECEIPT_CHAIN_SCHEMA,
            "entries": [],
        },
    )


def append_reuse_receipt(
    output_dir: str | Path,
    *,
    capsule_id: str,
    load_case_id: str,
    fingerprint: str,
    plugin_id: str,
    branches: tuple[str, ...],
    validator_source_sha256: str,
    origin_run: str,
    origin_plan_sha256: str,
    origin_load_case_sha256: str,
    origin_package_binding_sha256: str,
    current_plan_sha256: str,
    current_load_case_sha256: str,
    current_package_binding_sha256: str,
    artifacts: Mapping[str, Mapping[str, str]],
    origin_outer_audit: Mapping[str, Any],
    signed_by: str,
    reason: str,
) -> dict[str, Any]:
    """Append one sign-off receipt to the run's hash chain."""

    entries = load_reuse_receipts(output_dir)
    entry: dict[str, Any] = {
        "schema_version": REUSE_RECEIPT_SCHEMA,
        "capsule_id": capsule_id,
        "load_case_id": load_case_id,
        "fingerprint": fingerprint,
        "plugin_id": plugin_id,
        "branches": sorted(branches),
        "validator_source_sha256": validator_source_sha256,
        "origin_run": origin_run,
        "origin_plan_sha256": origin_plan_sha256,
        "origin_load_case_sha256": origin_load_case_sha256,
        "origin_package_binding_sha256": origin_package_binding_sha256,
        "current_plan_sha256": current_plan_sha256,
        "current_load_case_sha256": current_load_case_sha256,
        "current_package_binding_sha256": current_package_binding_sha256,
        "artifacts": {
            label: dict(binding) for label, binding in sorted(artifacts.items())
        },
        "origin_outer_audit": dict(origin_outer_audit),
        "signed_by": signed_by,
        "reason": reason,
        "signed_at": _datetime.datetime.now(
            _datetime.timezone.utc
        ).isoformat(),
        "previous_entry_sha256": (
            entries[-1]["entry_sha256"] if entries else None
        ),
    }
    entry["entry_sha256"] = _entry_digest(entry)
    write_json_atomic(
        _receipts_path(Path(output_dir)),
        {
            "schema_version": REUSE_RECEIPT_CHAIN_SCHEMA,
            "entries": [*entries, entry],
        },
    )
    return entry


def receipt_for_lane(
    output_dir: str | Path,
    *,
    capsule_id: str,
    load_case_id: str,
) -> dict[str, Any] | None:
    """Chain-verified receipt lookup for one lane; None when not signed."""

    for entry in load_reuse_receipts(output_dir):
        if (
            entry.get("capsule_id") == capsule_id
            and entry.get("load_case_id") == load_case_id
        ):
            return entry
    return None


def verify_lane_execution_root(
    *,
    workspace_root: Path,
    origin_run_dir: Path,
    capsule_id: str,
    load_case_id: str,
    fingerprint: str,
    plugin_id: str,
    branches: tuple[str, ...],
    validator_source_sha256: str,
    include_ideal_dynamics: bool,
    task_plugin_binding: Mapping[str, str] | None = None,
    extra_dropped_keys: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """Fully re-verify one already-executed lane inside an execution root.

    Every claim a sign-off later relies on is recomputed here from the
    origin files themselves — nothing is trusted from a reuse or promotion
    hint. Any inconsistency returns ``None`` (the lane executes normally);
    sign-off is an optimization and always fails open to recomputation,
    never to acceptance.
    """

    try:
        origin_plan_path = origin_run_dir / "plans" / f"{capsule_id}.json"
        origin_load_path = (
            origin_run_dir / "loads" / capsule_id / f"{load_case_id}.json"
        )
        origin_plan_document = read_json(origin_plan_path)
        origin_load_document = read_json(origin_load_path)
        if (
            lane_fingerprint(
                plan_document=origin_plan_document,
                load_case_document=origin_load_document,
                validator_source_sha256=validator_source_sha256,
                plugin_id=plugin_id,
                branches=branches,
                task_plugin_binding=task_plugin_binding,
                extra_dropped_keys=extra_dropped_keys,
            )
            != fingerprint
        ):
            return None
        origin_plan_sha256 = canonical_sha256(origin_plan_document)
        origin_load_sha256 = canonical_sha256(origin_load_document)
        origin_package_binding = str(
            origin_plan_document.get("package_binding_sha256") or ""
        )

        lane_path = (
            origin_run_dir
            / "results"
            / capsule_id
            / "lanes"
            / f"{load_case_id}.json"
        )
        lane = read_json(lane_path)
        expected_labels = set(LANE_SIGN_OFF_LABELS)
        if include_ideal_dynamics:
            expected_labels.update(LANE_SIGN_OFF_IDEAL_DYNAMICS_LABELS)
        artifacts = lane.get("artifacts")
        if (
            lane.get("schema_version")
            != "simulation_ready_cad_mechanism_load_lane_v1"
            or lane.get("status") != "PASS"
            or lane.get("completed_phase") != "comparison"
            or lane.get("capsule_id") != capsule_id
            or lane.get("load_case_id") != load_case_id
            or lane.get("plan_sha256") != origin_plan_sha256
            or lane.get("load_case_sha256") != origin_load_sha256
            or lane.get("package_binding_sha256") != origin_package_binding
            # Same fingerprint means same normalized inputs; results are
            # transferable only when the same validator source computed them.
            or lane.get("validator_source_sha256") != validator_source_sha256
            or not isinstance(artifacts, dict)
            or set(artifacts) != expected_labels
        ):
            return None

        verified_artifacts: dict[str, dict[str, str]] = {}
        for label in sorted(expected_labels):
            binding = artifacts.get(label)
            if not isinstance(binding, dict):
                return None
            origin_file = binding.get("file")
            expected_sha256 = binding.get("sha256")
            if not isinstance(origin_file, str) or not isinstance(
                expected_sha256, str
            ):
                return None
            path = (workspace_root / origin_file).resolve()
            if (
                not path.is_relative_to(origin_run_dir)
                or not path.is_file()
                or path.is_symlink()
                or sha256_file(path) != expected_sha256
            ):
                return None
            verified_artifacts[label] = {
                "origin_file": origin_file,
                "sha256": expected_sha256,
            }

        # The physical trace must carry the origin identity it claims.
        physical_path = (
            workspace_root / verified_artifacts["physical_trace"]["origin_file"]
        ).resolve()
        physical = read_trace_json(physical_path)
        if (
            physical.get("status") != "COMPLETED"
            or physical.get("branch") != "physical"
            or physical.get("capsule_id") != capsule_id
            or physical.get("load_case_id") != load_case_id
            or physical.get("plan_sha256") != origin_plan_sha256
            or physical.get("load_case_sha256") != origin_load_sha256
            or physical.get("package_binding_sha256")
            != origin_package_binding
            or physical.get("validator_source_sha256")
            != validator_source_sha256
        ):
            return None

        # Only results the origin run itself verified may be signed off:
        # its outer CAD/FCL preflight must record this exact trace as PASS.
        preflight = read_json(
            origin_run_dir / "results" / "physical_preflight.json"
        )
        rows = preflight.get("traces")
        matching = [
            row
            for row in rows
            if isinstance(row, dict)
            and row.get("capsule_id") == capsule_id
            and row.get("load_case_id") == load_case_id
            and row.get("trace")
            == verified_artifacts["physical_trace"]["origin_file"]
        ] if isinstance(rows, list) else []
        if (
            len(matching) != 1
            or matching[0].get("passed") is not True
            or matching[0].get("trace_sha256")
            != verified_artifacts["physical_trace"]["sha256"]
        ):
            return None

        # The adopted lane result itself travels with the artifacts.
        verified_artifacts["lane_result"] = {
            "origin_file": (
                lane_path.resolve().relative_to(workspace_root).as_posix()
            ),
            "sha256": sha256_file(lane_path),
        }
        return {
            "origin_run": (
                origin_run_dir.resolve()
                .relative_to(workspace_root)
                .as_posix()
            ),
            "origin_plan_sha256": origin_plan_sha256,
            "origin_load_case_sha256": origin_load_sha256,
            "origin_package_binding_sha256": origin_package_binding,
            "artifacts": verified_artifacts,
            "origin_outer_audit": {
                "trace": verified_artifacts["physical_trace"]["origin_file"],
                "trace_sha256": (
                    verified_artifacts["physical_trace"]["sha256"]
                ),
                "passed": True,
            },
        }
    except (OSError, ValueError, KeyError, TypeError):
        return None


def find_lane_sign_off_source(
    *,
    workspace_root: Path,
    output_dir: Path,
    capsule_id: str,
    load_case_id: str,
    fingerprint: str,
    plugin_id: str,
    branches: tuple[str, ...],
    validator_source_sha256: str,
    include_ideal_dynamics: bool,
    task_plugin_binding: Mapping[str, str] | None = None,
    extra_dropped_keys: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """Search sibling runs for a verified lane with this exact fingerprint.

    Only canonical ``<project>/runs/<run>`` layouts participate.  Chains of
    sign-offs resolve to the run that actually *executed* the physics: a
    sibling whose manifest says REUSED points at its recorded origin, and
    verification always happens against that origin's own files.
    """

    run_dir = Path(output_dir).resolve()
    runs_dir = run_dir.parent
    if runs_dir.name != "runs":
        return None
    workspace = Path(workspace_root).resolve()
    for sibling in sorted(runs_dir.iterdir(), reverse=True):
        if not sibling.is_dir() or sibling == run_dir:
            continue
        manifest_path = sibling / "results" / "reuse_manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError):
            continue
        lane_jobs = manifest.get("lane_jobs")
        if not isinstance(lane_jobs, list):
            continue
        for entry in lane_jobs:
            # The sibling manifest is only a routing hint (which run
            # executed this lane); its recorded fingerprint is
            # informational and may predate a fingerprint schema bump.
            # The load-bearing equality check is the recomputation from
            # the origin's own plan/load documents inside
            # ``verify_lane_execution_root``.
            if (
                not isinstance(entry, dict)
                or entry.get("capsule_id") != capsule_id
                or entry.get("load_case_id") != load_case_id
            ):
                continue
            if entry.get("action") == "EXECUTED":
                origin_run_dir = sibling
            elif entry.get("action") == "REUSED" and isinstance(
                entry.get("origin_run"), str
            ):
                origin_run_dir = (workspace / entry["origin_run"]).resolve()
                if origin_run_dir.parent != runs_dir:
                    # ``stage-replay --output`` supports any directory under
                    # the diagnostic root.  A later canonical run may point
                    # back to that explicitly promoted execution root, but no
                    # other non-canonical layout participates.
                    diagnostic_root = (
                        workspace / ".runtime_runs" / "_diagnostics"
                    ).resolve()
                    try:
                        origin_run_dir.relative_to(diagnostic_root)
                    except ValueError:
                        continue
            else:
                continue
            verified = verify_lane_execution_root(
                workspace_root=workspace,
                origin_run_dir=origin_run_dir,
                capsule_id=capsule_id,
                load_case_id=load_case_id,
                fingerprint=fingerprint,
                plugin_id=plugin_id,
                branches=branches,
                validator_source_sha256=validator_source_sha256,
                include_ideal_dynamics=include_ideal_dynamics,
                task_plugin_binding=task_plugin_binding,
                extra_dropped_keys=extra_dropped_keys,
            )
            if verified is not None:
                return verified
    return None


def solid_geometry_sha256(solid: Any) -> str:
    """Serialize one OCCT solid and hash its BRep geometry content."""

    from OCP.BRepTools import BRepTools

    handle, temp_name = tempfile.mkstemp(suffix=".brep")
    try:
        os.close(handle)
        BRepTools.Write_s(solid, temp_name)
        return sha256_file(Path(temp_name))
    finally:
        try:
            os.unlink(temp_name)
        except OSError:
            pass


class ContentAddressedMeshCache:
    """Project-scoped store for deterministic solid tessellations.

    Layout: ``<root>/<key[:2]>/<key>.json`` records ``{payload_sha256}``
    next to ``<key>.stl`` payload bytes.  Fetch verifies the payload hash
    and treats any inconsistency as a miss (removing the bad entry), so a
    damaged cache can only cost time, never correctness.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.hits = 0
        self.misses = 0
        self.stores = 0

    @staticmethod
    def solid_key(solid: Any, *, mesh_parameters: Mapping[str, Any]) -> str:
        """Key a solid by its BRep content before any triangulation."""
        return canonical_sha256(
            {
                "schema_version": MESH_CACHE_SCHEMA,
                "geometry_sha256": solid_geometry_sha256(solid),
                "mesh_parameters": dict(mesh_parameters),
            }
        )

    def _record_path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def _payload_path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.stl"

    def fetch(self, key: str, destination: Path) -> bool:
        record_path = self._record_path(key)
        payload_path = self._payload_path(key)
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            expected = str(record["payload_sha256"])
            if (
                record.get("schema_version") != MESH_CACHE_SCHEMA
                or sha256_file(payload_path) != expected
            ):
                raise ValueError("cache entry failed verification")
        except (OSError, ValueError, KeyError, TypeError):
            for stale in (record_path, payload_path):
                try:
                    stale.unlink()
                except OSError:
                    pass
            self.misses += 1
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload_path.read_bytes())
        self.hits += 1
        return True

    def store(self, key: str, source: Path) -> None:
        try:
            payload_path = self._payload_path(key)
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            handle, temp_name = tempfile.mkstemp(
                dir=str(payload_path.parent), suffix=".tmp"
            )
            with os.fdopen(handle, "wb") as stream:
                stream.write(source.read_bytes())
            os.replace(temp_name, payload_path)
            write_json_atomic(
                self._record_path(key),
                {
                    "schema_version": MESH_CACHE_SCHEMA,
                    "payload_sha256": sha256_file(payload_path),
                },
            )
            self.stores += 1
        except OSError:
            # The cache is an optimization; failing to persist an entry
            # must never fail the validation run.
            pass

    def stats(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
        }


def mesh_cache_for_run(output_dir: Path) -> ContentAddressedMeshCache | None:
    """Locate the project-scoped mesh cache for one run directory.

    Canonical layout is ``<project>/runs/<run>``; the cache lives beside
    the runs so every revision of the project shares it.  Runs outside the
    canonical layout (unit-test scratch directories) get no cache.
    """

    run_dir = Path(output_dir).resolve()
    runs_dir = run_dir.parent
    if runs_dir.name != "runs":
        return None
    return ContentAddressedMeshCache(
        runs_dir.parent / "reuse_cache" / "mesh_v1"
    )
