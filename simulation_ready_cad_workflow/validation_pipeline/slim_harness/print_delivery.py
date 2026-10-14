"""Executed, task-bound print delivery evidence; no mechanical or document verdict.

The existing alarm F7 verifier is retained unchanged. A supervisor supplies the
print/export mapping outside the author directory; the CAD worker checks it and
the independent reviewer remains responsible for its relation to frozen criteria.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import shutil
import sys
import time
from urllib.parse import unquote, urlparse

from ..runtime.bounded_subprocess import run_bounded_subprocess
from .static_delivery import (
    F7, VerifiedStaticEvidence, binding, check_binding, digest, read, write,
    load_registered_static_evidence as load_alarm_evidence,
)

VERIFIER_ID = "print_delivery_v1"
REGISTRATION_FILE = "print-delivery-registration.json"


@dataclass(frozen=True)
class PrintStaticEvidence(VerifiedStaticEvidence):
    criterion_json: str
    coverage_mode: str
    static_scope: str = ""


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def applies(evidence, criterion):
    if isinstance(evidence, PrintStaticEvidence):
        return evidence.criterion_json == canonical(criterion)
    return isinstance(evidence, VerifiedStaticEvidence) and criterion == F7


def is_static_delivery_criterion(criterion, static_evidence=()):
    return criterion == F7 or any(
        applies(row, criterion) and row.coverage_mode == "static_only"
        for row in static_evidence if isinstance(row, PrintStaticEvidence)
    )


def has_static_evidence(criterion, static_evidence):
    return any(applies(row, criterion) for row in static_evidence)


def implementation_binding():
    from . import geometry, source_queries, static_delivery_worker, source_mesh_refinement, source_analytic_material
    from ..runtime import bounded_subprocess, process_containment, system_probe
    return binding([__file__, Path(__file__).with_name("print_delivery_worker.py"),
                    geometry.__file__, source_mesh_refinement.__file__, source_analytic_material.__file__,
                    source_queries.__file__, static_delivery_worker.__file__,
                    bounded_subprocess.__file__, process_containment.__file__, system_probe.__file__])


def document_images(design, documents):
    """Bind the local image files used by the delivered HTML assembly guides."""
    images = set()
    class Images(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag.lower() == "img":
                value = dict(attrs).get("src", "")
                parsed = urlparse(value)
                if value and not parsed.scheme and not parsed.netloc:
                    path = (self.document.parent / unquote(parsed.path)).resolve()
                    if not path.is_relative_to(design):
                        raise ValueError("Assembly guide image is outside its bound design")
                    images.add(path.relative_to(design).as_posix())
    for relative in documents:
        path = design / relative
        if path.suffix.lower() in {".html", ".htm"}:
            parser = Images()
            parser.document = path
            parser.feed(path.read_text(encoding="utf-8-sig"))
    return images


def validate_request(project, design, request_path):
    project, design, request_path = map(lambda p: Path(p).resolve(), (project, design, request_path))
    if request_path.is_relative_to(design):
        raise ValueError("The author design cannot own the verifier request")
    request = read(request_path)
    if request["verifier_id"] != VERIFIER_ID or request["scene_path"] != str(design / "scene.json"):
        raise ValueError("Wrong verifier or scene binding")
    criteria = read(project / "criteria.json")
    lock = read(project / "criteria.lock.json")
    if (lock["criteria_sha256"] != digest(project / "criteria.json")
            or lock["task_sha256"] != digest(project / "task.md")
            or criteria["task_text"] != (project / "task.md").read_text(encoding="utf-8")):
        raise ValueError("Frozen task/criteria binding changed")
    declarations = request["criteria"]
    if not declarations or len({row["criterion"]["id"] for row in declarations}) != len(declarations):
        raise ValueError("Static criterion declarations must be nonempty and unique")
    for row in declarations:
        if row["criterion"] not in criteria["criteria"] or row["coverage_mode"] not in {"static_only", "supplemental"}:
            raise ValueError("Static coverage must bind an exact frozen criterion and explicit mode")
        if not row.get("scope"):
            raise ValueError("State the static scope for independent adequacy review")
    files = request["source_files"]
    if not files or len(set(files)) != len(files):
        raise ValueError("Source snapshot must be nonempty and unique")
    for relative in files:
        if Path(relative).is_absolute() or not (design / relative).resolve().is_relative_to(design):
            raise ValueError("Source path escapes the current design")
    scene = read(design / "scene.json")
    required = {"scene.json", request["assembly_step"], request["assembly_glb"], request["quantity_file"],
                *request["documents"], *scene.get("authoring_sources", []),
                *(body["source"] for body in scene["bodies"]),
                *(part["source"] for part in request["parts"]), *(part["stl"] for part in request["parts"]),
                *(plate["stl"] for plate in request["plates"]),
                *document_images(design, request["documents"])}
    if not required <= set(files):
        raise ValueError("Incomplete print source snapshot: " + str(sorted(required - set(files))))
    if not request["build_steps"] or not set(request["build_inputs"]) <= set(files):
        raise ValueError("Rebuild requires bound inputs and executed source commands")
    if any(not step or step[0] not in request["build_inputs"] for step in request["build_steps"]):
        raise ValueError("Each build step must execute a bound delivered source")
    generated = {request["assembly_step"], *(part["source"] for part in request["parts"]),
                 *(part["stl"] for part in request["parts"])}
    if generated.intersection(request["build_inputs"]):
        raise ValueError("Fresh assembly, part STEP sources and STLs cannot be copied into the rebuild")
    sources = binding(design / relative for relative in files)
    if request["source_sha256"] != {Path(p).relative_to(design).as_posix(): sha for p, sha in sources.items()}:
        raise ValueError("Current source snapshot changed")
    return request, sources, binding(project / name for name in ("criteria.json", "criteria.lock.json", "task.md"))


def execute(project, design, request_path, out, cad_python, *, register=False):
    project, design, request_path, out = map(lambda p: Path(p).resolve(), (project, design, request_path, out))
    request, sources, criteria = validate_request(project, design, request_path)
    if out.exists() or out.is_relative_to(design):
        raise ValueError("Preserve existing evidence and keep execution outside the author design")
    out.mkdir(parents=True)
    started = time.monotonic()
    receipt = {"verifier_id": VERIFIER_ID, "scene_path": str(design / "scene.json"),
               "project_path": str(project), "request_path": str(request_path),
               "request_sha256": digest(request_path), "source_bindings": sources,
               "criteria_bindings": criteria, "implementation_bindings": implementation_binding(),
               "started_at": datetime.now(timezone.utc).isoformat(), "stages": [], "outcome": "incomplete",
               "scope": "Executed source rebuild and print/export measurements only. Independent document, assembly and criterion-coverage review required. No mechanical or real-print acceptance."}
    env = dict(os.environ)
    env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3]) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        clone = out / "rebuilt_design"
        clone.mkdir()
        for relative in request["build_inputs"]:
            destination = clone / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(design / relative, destination)

        def run_stage(name, command, cwd):
            result = run_bounded_subprocess(command, cwd=cwd, env=env, timeout_seconds=600, max_rss_bytes="auto")
            stdout, stderr = out / f"{name}.stdout.log", out / f"{name}.stderr.log"
            stdout.write_bytes(result.stdout)
            stderr.write_bytes(result.stderr)
            receipt["stages"].append({"name": name, "command": list(map(str, command)),
                                      "returncode": result.returncode, "logs": binding([stdout, stderr])})
            if result.returncode:
                raise ValueError(f"{name} failed; see {stderr}")

        for index, step in enumerate(request["build_steps"]):
            run_stage(f"build_{index + 1}", [str(cad_python), "-X", "utf8", str(clone / step[0]), *step[1:]], clone)
        payload = out / "inspection-request.json"
        inspection_path = out / "inspection.json"
        write(payload, {"design": str(design), "rebuilt_design": str(clone), "request": request,
                        "output": str(inspection_path)})
        run_stage("inspect_print_delivery", [str(cad_python), "-X", "utf8", str(Path(__file__).with_name("print_delivery_worker.py")), str(payload)], out)
        inspection = read(inspection_path)
        if inspection.get("errors") or not inspection.get("completed"):
            raise ValueError("Print inspection incomplete: " + str(inspection.get("errors")))
        if not all(check_binding(group) for group in (sources, criteria, receipt["implementation_bindings"], inspection["runtime_bindings"])):
            raise ValueError("Source, criteria or implementation changed during execution")
        receipt.update(outcome="completed", inspection_path=str(inspection_path),
                       runtime_bindings=inspection["runtime_bindings"],
                       artifact_bindings=binding([payload, inspection_path, *(clone / path for path in inspection["rebuilt_files"])]))
    except Exception as exc:
        receipt.update(outcome="error", error=f"{type(exc).__name__}: {exc}")
    receipt["wall_seconds"] = time.monotonic() - started
    receipt["receipt_id"] = VERIFIER_ID + ":" + __import__("hashlib").sha256(canonical({"request": receipt["request_sha256"], "started": receipt["started_at"]}).encode()).hexdigest()[:24]
    receipt_path = out / "receipt.json"
    write(receipt_path, receipt)
    if register and receipt["outcome"] == "completed":
        register_existing(project, design, request_path, receipt_path)
    return receipt


def _load_entry(project, scene_path, entry):
    design = scene_path.parent
    from ..snapshot_paths import resolve, binding_paths, implementation_matches
    workspace = project.parents[2]
    mapped = lambda value: resolve(workspace, value)
    bindings = lambda value: binding_paths(workspace, value)
    request_path, receipt_path = mapped(entry['request_path']), mapped(entry['receipt_path'])
    if receipt_path.is_relative_to(design) or digest(receipt_path) != entry["receipt_sha256"]:
        raise ValueError("Receipt location or digest changed")
    request, sources, criteria = validate_request(project, design, request_path)
    receipt = read(receipt_path)
    if (receipt["verifier_id"] != VERIFIER_ID or receipt["outcome"] != "completed"
            or mapped(receipt['scene_path']) != scene_path or mapped(receipt['project_path']) != project
            or receipt["request_sha256"] != digest(request_path) or bindings(receipt["source_bindings"]) != sources
            or bindings(receipt["criteria_bindings"]) != criteria or not implementation_matches(workspace, receipt["implementation_bindings"], implementation_binding())):
        raise ValueError("Receipt no longer binds the current task, design and executed verifier")
    if not all(check_binding(bindings(receipt[key])) for key in ("runtime_bindings", "artifact_bindings")):
        raise ValueError("Executed runtime or output evidence changed")
    expected = [f"build_{n + 1}" for n in range(len(request["build_steps"]))] + ["inspect_print_delivery"]
    if ([stage["name"] for stage in receipt["stages"]] != expected
            or any(stage["returncode"] or not check_binding(bindings(stage["logs"])) for stage in receipt["stages"])):
        raise ValueError("Executed stages or logs are incomplete")
    inspection = read(mapped(receipt['inspection_path']))
    if not inspection.get("completed") or inspection.get("errors"):
        raise ValueError("No completed print inspection")
    return tuple(PrintStaticEvidence(
        receipt["receipt_id"] + ":" + row["criterion"]["id"], str(receipt_path), entry["receipt_sha256"],
        str(scene_path), digest(project / "criteria.json"),
        (str(receipt_path), str(mapped(receipt['inspection_path'])), *(str(design / path) for path in request["documents"])),
        canonical(row["criterion"]), row["coverage_mode"], row["scope"],
    ) for row in request["criteria"])


def register_existing(project, design, request_path, receipt_path):
    project, design, request_path, receipt_path = map(lambda p: Path(p).resolve(), (project, design, request_path, receipt_path))
    entry = {"verifier_id": VERIFIER_ID, "request_path": str(request_path), "receipt_path": str(receipt_path), "receipt_sha256": digest(receipt_path)}
    evidence = _load_entry(project, design / "scene.json", entry)
    write(project / REGISTRATION_FILE, entry)
    return evidence


def load_registered_static_evidence(project, scene_path):
    project, scene_path = Path(project).resolve(), Path(scene_path).resolve()
    evidence, errors = load_alarm_evidence(project, scene_path)
    from .step_delivery import load_registered_step_evidence
    step_evidence,step_errors=load_registered_step_evidence(project,scene_path)
    evidence,errors=(*evidence,*step_evidence),[*errors,*step_errors]
    path = project / REGISTRATION_FILE
    if not path.is_file():
        return evidence, errors
    try:
        return (*evidence, *_load_entry(project, scene_path, read(path))), errors
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return evidence, [*errors, "Print delivery receipt rejected: " + str(exc)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "design", "request"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--out")
    parser.add_argument("--cad-python")
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--register-existing")
    args = parser.parse_args()
    if args.register_existing:
        evidence = register_existing(args.project, args.design, args.request, args.register_existing)
        print(json.dumps({"receipt_ids": [row.receipt_id for row in evidence], "outcome": "registered"}))
        return 0
    if not args.out or not args.cad_python:
        parser.error("Execution requires --out and --cad-python")
    receipt = execute(args.project, args.design, args.request, args.out, args.cad_python, register=args.register)
    print(json.dumps({key: receipt[key] for key in ("outcome", "receipt_id", "wall_seconds")}, ensure_ascii=False))
    return 0 if receipt["outcome"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
