"""Program-owned, executed static delivery evidence for the unchanged alarm-clock F7.

This module never accepts an author supplied success flag. A reviewer may cite
only a receipt loaded from the project registration after its bindings validate.
It does not certify documentation semantics or any mechanical function.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

from simulation_ready_cad_workflow.validation_pipeline.runtime import bounded_subprocess
from simulation_ready_cad_workflow.validation_pipeline.runtime import process_containment, system_probe

VERIFIER_ID = "alarm_clock_f7_static_delivery_v1"
REGISTRATION_FILE = "static-delivery-registration.json"
F7 = {
    "id": "F7", "kind": "function",
    "statement": "交付建模源文件、完整闹钟的 STEP 装配、零件及连接关系说明，以及简要操作说明。",
    "source_quote": "请交付建模源文件、STEP 装配、零件及连接关系说明，以及简要操作说明。",
    "verification": "确认源文件和 STEP 装配可用相应工具打开；核对说明与装配中的零件及连接关系相符，操作说明涵盖上弦、调整当前时间、设置闹铃和关闭铃声。",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def binding(paths):
    return {str(Path(p).resolve()): digest(p) for p in sorted(set(map(str, paths)))}


def check_binding(expected):
    return bool(expected) and all(Path(p).is_file() and digest(p) == sha for p, sha in expected.items())


def implementation_binding():
    return binding([__file__, Path(__file__).with_name("static_delivery_worker.py"),
                    bounded_subprocess.__file__, process_containment.__file__, system_probe.__file__])


def is_static_delivery_criterion(criterion):
    return criterion == F7


@dataclass(frozen=True)
class VerifiedStaticEvidence:
    receipt_id: str
    receipt_path: str
    receipt_sha256: str
    scene_path: str
    criteria_sha256: str
    evidence_paths: tuple[str, ...]


def source_binding(design, request):
    files = request["source_files"]
    if not files or len(files) != len(set(files)):
        raise ValueError("source_files must be a nonempty unique bound snapshot")
    for relative in files:
        if not (design / relative).resolve().is_relative_to(design):
            raise ValueError("Source snapshot path escapes the design")
    scene = read(design / "scene.json")
    required = {"scene.json", *scene.get("authoring_sources", []),
                *(body["source"] for body in scene["bodies"]),
                request["assembly"], request["body_manifest"], request["joint_manifest"],
                request["parts_csv"], *request["documents"],
                *(step[0] for step in request["build_steps"])}
    if not required <= set(files):
        raise ValueError("Incomplete source snapshot: " + str(sorted(required - set(files))))
    return binding(design / relative for relative in files)


def validate_request(project, design, tooling, request_path):
    request = read(request_path)
    if request["verifier_id"] != VERIFIER_ID:
        raise ValueError("Unregistered static verifier")
    criteria = read(project / "criteria.json")
    if [c for c in criteria["criteria"] if c["id"] == "F7"] != [F7]:
        raise ValueError("This verifier applies only to the exact frozen alarm-clock F7")
    if criteria["task_text"] != (project / "task.md").read_text(encoding="utf-8"):
        raise ValueError("Frozen task text changed")
    lock = read(project / "criteria.lock.json")
    if lock["criteria_sha256"] != digest(project / "criteria.json") or lock["task_sha256"] != digest(project / "task.md"):
        raise ValueError("Frozen criteria lock mismatch")
    sources = source_binding(design, request)
    if request.get("expected_source_sha256") != {Path(p).relative_to(design).as_posix(): v for p, v in sources.items()}:
        raise ValueError("Request does not bind the exact current source snapshot")
    tools = binding(tooling / path for path in request["tooling_files"])
    if request["expected_tooling_sha256"] != {Path(p).relative_to(tooling).as_posix(): v for p, v in tools.items()}:
        raise ValueError("Modeling helper changed")
    if request["criteria_sha256"] != digest(project / "criteria.json"):
        raise ValueError("Request criteria binding changed")
    output = Path(request["fresh_output_dir"])
    if output.is_absolute() or not (design / output).resolve().is_relative_to(design) or output == Path("."):
        raise ValueError("Fresh modeling output must be a proper design subdirectory")
    if not Path(request["assembly"]).is_relative_to(output):
        raise ValueError("The complete assembly must be produced in the fresh output directory")
    if not request["build_steps"] or any(not row or row[0] not in request["source_files"] for row in request["build_steps"]):
        raise ValueError("Every build step must execute a bound delivered source")
    return request, sources, tools


def execute(project, design, tooling, request_path, cad_python, out, *, register=False):
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    project, design, tooling, request_path, out = map(lambda p: Path(p).resolve(), (project, design, tooling, request_path, out))
    if request_path.is_relative_to(design) or out.is_relative_to(design):
        raise ValueError("The Harness request and execution evidence must be outside the author design")
    if out.exists():
        raise ValueError("Preserve existing execution evidence; choose a new output directory")
    request, sources, tool_sources = validate_request(project, design, tooling, request_path)
    out.mkdir(parents=True)
    criteria_files = binding(project / name for name in ("criteria.json", "criteria.lock.json", "task.md"))
    receipt = {
        "schema_version": "slim_static_delivery_receipt_v1", "verifier_id": VERIFIER_ID,
        "criterion": F7, "scene_path": str(design / "scene.json"), "project_path": str(project),
        "request_path": str(request_path), "request_sha256": digest(request_path),
        "source_bindings": sources, "tooling_bindings": tool_sources, "criteria_bindings": criteria_files,
        "implementation_bindings": implementation_binding(), "cad_python": str(Path(cad_python).resolve()),
        "started_at": started_at, "stages": [], "outcome": "incomplete",
        "scope": "Executed source rebuild, full STEP opening and inventory/ID correspondence; independent F7 documentation and semantic observation still required. No mechanical outcome.",
    }
    receipt_path = out / "receipt.json"
    clone = out / "build_design"
    try:
        clone.mkdir()
        excluded = Path(request["fresh_output_dir"])
        for relative in request["source_files"]:
            if Path(relative).is_relative_to(excluded):
                continue
            destination = clone / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(design / relative, destination)
        if (clone / excluded).exists():
            raise ValueError("Fresh output was unexpectedly present before source execution")
        parsed = []
        for relative in request["source_files"]:
            if Path(relative).suffix == ".py":
                ast.parse((design / relative).read_text(encoding="utf-8-sig"), filename=relative)
                parsed.append(relative)
        receipt["source_parse"] = {"files": parsed, "scope": "Syntax only; not modeling execution evidence"}
        env = dict(os.environ)
        env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        env["PYTHONPATH"] = str(tooling) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        def run_stage(name, command, cwd):
            stage_started = time.monotonic()
            row = {"name": name, "command": list(map(str, command)), "cwd": str(cwd), "returncode": None}
            receipt["stages"].append(row)
            try:
                result = bounded_subprocess.run_bounded_subprocess(command, cwd=cwd, env=env,
                    timeout_seconds=request["timeout_seconds"], max_rss_bytes=request["max_rss_bytes"])
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                raise RuntimeError(f"{name} interrupted: {exc}") from exc
            finally:
                row["wall_seconds"] = time.monotonic() - stage_started
            stdout, stderr = out / (name + ".stdout.log"), out / (name + ".stderr.log")
            stdout.write_bytes(result.stdout)
            stderr.write_bytes(result.stderr)
            row.update(returncode=result.returncode, logs=binding([stdout, stderr]))
            if result.returncode:
                raise RuntimeError(f"{name} failed with exit code {result.returncode}; see {stderr}")

        for index, step in enumerate(request["build_steps"]):
            run_stage(f"build_{index + 1}", [receipt["cad_python"], "-X", "utf8", str(clone / step[0]), *step[1:]], clone)
        generated = clone / request["assembly"]
        if not generated.is_file() or generated.stat().st_size == 0:
            raise ValueError("Source processes produced no fresh complete STEP assembly")
        worker_input = out / "inspection-request.json"
        worker_output = out / "inspection.json"
        write(worker_input, {"design": str(design), "rebuilt_design": str(clone), "request": request, "output": str(worker_output)})
        run_stage("inspect_step_and_manifests", [receipt["cad_python"], "-X", "utf8", str(Path(__file__).with_name("static_delivery_worker.py")), str(worker_input)], out)
        inspection = read(worker_output)
        if inspection.get("errors") or not all(inspection.get(k) for k in ("original_assembly", "rebuilt_assembly", "body_correspondence", "runtime_bindings")):
            raise ValueError("Static inspection did not complete all required evidence: " + str(inspection.get("errors")))
        if not all(check_binding(b) for b in (sources, tool_sources, criteria_files, receipt["implementation_bindings"], inspection["runtime_bindings"])):
            raise ValueError("Source, criteria, tooling or implementation changed during execution")
        receipt["inspection_path"] = str(worker_output)
        receipt["runtime_bindings"] = inspection["runtime_bindings"]
        artifacts = [worker_input, worker_output, generated, clone / "scene.json", clone / request["body_manifest"],
                     clone / request["joint_manifest"], clone / request["parts_csv"]]
        artifacts.extend(clone / body["source"] for body in read(clone / "scene.json")["bodies"])
        receipt["artifact_bindings"] = binding(artifacts)
        receipt["outcome"] = "completed"
    except Exception as exc:
        receipt["outcome"] = "error"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    receipt["wall_seconds"] = time.monotonic() - started
    receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
    identity = {key: receipt[key] for key in ("request_sha256", "source_bindings", "criteria_bindings", "implementation_bindings", "started_at")}
    receipt["receipt_id"] = VERIFIER_ID + ":" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    write(receipt_path, receipt)
    if register and receipt["outcome"] == "completed":
        write(project / REGISTRATION_FILE, {"schema_version": "slim_static_delivery_registration_v1", "verifier_id": VERIFIER_ID,
              "receipt_path": str(receipt_path), "receipt_sha256": digest(receipt_path), "request_path": str(request_path), "tooling_root": str(tooling)})
    return receipt


def load_registered_static_evidence(project, scene_path):
    """Return only checked program receipts; stale/missing evidence never grants coverage."""
    project, scene_path = Path(project).resolve(), Path(scene_path).resolve()
    criteria = read(project / "criteria.json")
    if not any(is_static_delivery_criterion(row) for row in criteria["criteria"]):
        return (), []
    registration = project / REGISTRATION_FILE
    if not registration.is_file():
        return (), ["F7 has no registered static execution receipt"]
    try:
        return _load_static_entry(project, scene_path, read(registration))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return (), ["F7 static receipt rejected: " + str(exc)]


def _load_static_entry(project, scene_path, entry):
    try:
        from ..snapshot_paths import resolve, binding_paths, implementation_matches
        workspace = project.parents[2]
        mapped = lambda value: resolve(workspace, value)
        bindings = lambda value: binding_paths(workspace, value)
        receipt_path = mapped(entry['receipt_path'])
        if receipt_path.is_relative_to(scene_path.parent) or mapped(entry['request_path']).is_relative_to(scene_path.parent):
            raise ValueError("Author design cannot own the receipt or verifier request")
        if entry["verifier_id"] != VERIFIER_ID or digest(receipt_path) != entry["receipt_sha256"]:
            raise ValueError("Registered receipt digest or verifier changed")
        receipt = read(receipt_path)
        request, sources, tool_sources = validate_request(project, scene_path.parent, mapped(entry['tooling_root']), mapped(entry['request_path']))
        if receipt["verifier_id"] != VERIFIER_ID or receipt["criterion"] != F7 or receipt["outcome"] != "completed":
            raise ValueError("No completed registered F7 execution")
        if mapped(receipt['scene_path']) != scene_path or mapped(receipt['project_path']) != project:
            raise ValueError("Receipt belongs to another source revision")
        if receipt["request_sha256"] != digest(mapped(entry['request_path'])) or bindings(receipt["source_bindings"]) != sources or bindings(receipt["tooling_bindings"]) != tool_sources:
            raise ValueError("Request or source snapshot changed")
        if not implementation_matches(workspace, receipt['implementation_bindings'], implementation_binding()):
            raise ValueError("Static execution implementation changed")
        for key in ("criteria_bindings", "runtime_bindings", "artifact_bindings"):
            if not check_binding(bindings(receipt[key])):
                raise ValueError(key + " changed or missing")
        expected_stages = [f"build_{n + 1}" for n in range(len(request["build_steps"]))] + ["inspect_step_and_manifests"]
        if [row["name"] for row in receipt["stages"]] != expected_stages or any(row["returncode"] != 0 or not check_binding(bindings(row["logs"])) for row in receipt["stages"]):
            raise ValueError("Required executed stages or logs are missing")
        inspection = read(mapped(receipt['inspection_path']))
        if inspection["errors"] or not all(inspection.get(k) for k in ("original_assembly", "rebuilt_assembly", "body_correspondence", "runtime_bindings")):
            raise ValueError("Incomplete STEP or manifest execution evidence")
        files = (str(receipt_path), str(mapped(receipt['inspection_path'])), *(str(scene_path.parent / path) for path in request["documents"]))
        return (VerifiedStaticEvidence(receipt["receipt_id"], str(receipt_path), entry["receipt_sha256"], str(scene_path), digest(project / "criteria.json"), tuple(files)),), []
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return (), ["F7 static receipt rejected: " + str(exc)]


def register_existing_receipt(project, design, tooling, request_path, receipt_path):
    """Explicit post-review registration; reuse the very same current-evidence checks."""
    project, design, tooling, request_path, receipt_path = map(lambda p: Path(p).resolve(),
        (project, design, tooling, request_path, receipt_path))
    entry = {"schema_version": "slim_static_delivery_registration_v1", "verifier_id": VERIFIER_ID,
        "receipt_path": str(receipt_path), "receipt_sha256": digest(receipt_path),
        "request_path": str(request_path), "tooling_root": str(tooling)}
    evidence, errors = _load_static_entry(project, design / "scene.json", entry)
    if errors or not evidence:
        raise ValueError("Cannot register stale or incomplete execution evidence: " + "; ".join(errors))
    write(project / REGISTRATION_FILE, entry)
    return evidence[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("project", "design", "tooling", "request"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--cad-python")
    parser.add_argument("--out")
    parser.add_argument("--register", action="store_true", help="Register completed evidence in the project; never register an error receipt")
    parser.add_argument("--register-existing", metavar="RECEIPT", help="After independent review, revalidate and register an existing completed receipt without rebuilding")
    args = parser.parse_args()
    if args.register_existing:
        if args.out or args.cad_python or args.register:
            parser.error("--register-existing cannot be combined with new-execution options")
        evidence = register_existing_receipt(args.project, args.design, args.tooling, args.request, args.register_existing)
        print(json.dumps({"receipt_id": evidence.receipt_id, "outcome": "registered"}, ensure_ascii=False))
        return 0
    if not args.cad_python or not args.out:
        parser.error("New execution requires --cad-python and --out")
    result = execute(args.project, args.design, args.tooling, args.request, args.cad_python, args.out, register=args.register)
    print(json.dumps({key: result[key] for key in ("receipt_id", "outcome", "wall_seconds")}, ensure_ascii=False))
    return 0 if result["outcome"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
