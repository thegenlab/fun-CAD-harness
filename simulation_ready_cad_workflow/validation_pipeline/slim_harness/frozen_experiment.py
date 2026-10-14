"""Researcher-supplied tests use the same review/physics loop as generated tests.

Adapted from the earlier native experiment runner's frozen experiment ownership, using Slim's declarative
test interface rather than executing project Python inside the physical worker.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from ..mechanism.io import write_json_atomic
from .project_isolation import CODE_ROOT
from .trace_storage import file_digest


def validate_experiment(value, task):
    if value.get("schema_version") != "slim_frozen_experiment_v1":
        raise ValueError("Expected slim_frozen_experiment_v1")
    if value.get("task_text") != task:
        raise ValueError("Frozen experiment must match the original task exactly")
    if value.get("units") not in {"m", "mm"}:
        raise ValueError("Frozen experiment requires units m or mm")
    criteria = value.get("criteria", [])
    ids = [row.get("id") for row in criteria]
    if (not ids or any(not isinstance(i, str) or not i for i in ids)
            or len(ids) != len(set(ids))):
        raise ValueError("Frozen criteria require unique IDs")
    for row in criteria:
        if (row.get("kind") not in {"appearance", "function"}
                or not row.get("statement") or not row.get("verification")
                or not row.get("source_quote") or row["source_quote"] not in task):
            raise ValueError("Frozen criteria require statements, verification and exact task quotes")
    if not isinstance(value.get("tests"), list) or not value["tests"]:
        raise ValueError("Frozen experiment requires executable tests")
    if not value.get("reviewer_summary"):
        raise ValueError("Frozen experiment requires a researcher review summary")
    return value


def install_experiment(project: Path, source: Path):
    task = (project / "task.md").read_text(encoding="utf-8")
    value = validate_experiment(json.loads(source.read_text(encoding="utf-8-sig")), task)
    target = project / "frozen_experiment.json"
    write_json_atomic(target, value)
    return file_digest(target)


def load_experiment(project: Path):
    state = json.loads((project / "workflow.json").read_text(encoding="utf-8"))
    expected = state["request"].get("frozen_experiment_sha256")
    if expected is None:
        return None
    target = project / "frozen_experiment.json"
    if not target.is_file() or file_digest(target) != expected:
        raise ValueError("Frozen experiment changed; preserve this experiment and start a new one")
    return validate_experiment(json.loads(target.read_text(encoding="utf-8")),
                               (project / "task.md").read_text(encoding="utf-8"))


def bind_scene(scene, experiment):
    """Supply absent test fields; never accept a changed frozen test or environment."""
    if experiment is None:
        return scene
    value = copy.deepcopy(scene)
    for name, expected in (("units", experiment["units"]),
                           ("gravity_m_s2", experiment.get("gravity_m_s2", [0, -9.81, 0])),
                           ("tests", experiment["tests"])):
        if name in value and value[name] != expected:
            raise ValueError(f"Frozen experiment field {name} cannot be changed by the design author")
        value[name] = copy.deepcopy(expected)
    return value


def author_instructions(project: Path):
    experiment = load_experiment(project)
    workspace=project.parent.parent.parent
    draft=("You may work continuously within this author turn: build named parts, preview, check, "
           "and run development tests in any useful order. These draft results never certify the machine. "
           "Use simulate --test for one complete test when feedback is useful. Development simulation returns "
           "native physics first; full saved-motion geometry audit is deferred to formal validation. "
           "Inspect the functional result. Use the audit command on an existing request/trace "
           "when you need to check saved motion without rerunning physics. Do not wait for final "
           "certification in this author turn: once the source and test delivery are complete, return ready; "
           "the formal workflow reuses matching physics and performs source audit and code validation. "
           "Only changes to simulation tasks receive time-limited review advice; there is no final model review. "
           f"Read {CODE_ROOT/'simulation_ready_cad_workflow/validation_pipeline/slim_harness/AUTHOR_TOOLS.md'} "
           "for the Asset/Part helper and direct build/preview/check/simulate/audit commands. "
           f"Use --workspace \"{workspace}\" and --project \"{project}\" on draft commands. "
           "When listing, searching, or reading files, stay inside this project directory. "
           "Do not scan `.runtime_runs/projects/` for other machines, and do not open another "
           "project's model.py, scene.json, construction plan, or design tree. "
           "Preserve failed attempts; return ready only when the normal scene.json delivery is complete. ")
    if experiment is None:
        return draft+"Propose task-specific tests; independent review will check their coverage and physical meaning. "
    return (draft+"This run uses researcher-frozen tests. Keep the body/joint IDs required by the tests. "
            "Do not change tests, loads, durations, units, gravity or acceptance criteria. "
            "You may omit scene.tests; the program supplies the frozen tests. "
            f"Read {project / 'frozen_experiment.json'}. ")
