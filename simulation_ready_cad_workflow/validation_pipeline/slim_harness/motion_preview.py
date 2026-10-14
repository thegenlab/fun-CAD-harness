"""Leave a playable motion of the design when a run ends without one.

A run can end (budget used up, partial delivery, final evaluation) without any
playable simulation: tests stopped before their recorded poses were exported,
or every simulation diverging at once. People still want to see their machine
move, so after the run the harness exports recorded poses it has not replayed
yet and, failing that, lets the newest prepared design settle under gravity for
a moment. These motions are previews marked UNVERIFIED_PREFIX, never test
evidence or acceptance.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

from ..mechanism.io import write_json_atomic

PLAYABLE_FRAMES = 2
SETTLE_SECONDS = 1.5
SETTLE_STEP_SECONDS = 0.001
REPLAY_TIMEOUT_SECONDS = 120
SETTLE_TIMEOUT_SECONDS = 240
PARTIAL_SCHEMA = "slim_partial_motion_preview_v1"
TEMPLATE_TASKS = __package__ + ".template_tasks"


def _read(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _frames(motion: dict) -> int:
    try:
        return int(motion.get("motion_frame_count") or 0)
    except (TypeError, ValueError):
        return 0


def revision_motions(run: Path) -> list[dict]:
    """A revision's motions, including the preview motions merged by the dashboard."""
    preview = _read(run / "slim_evidence" / "preview.json")
    partial = _read(run / "slim_evidence" / "partial-motion-preview.json")
    motions = [item for item in preview.get("motions") or [] if isinstance(item, dict)]
    if partial.get("schema_version") == PARTIAL_SCHEMA and partial.get("input_version") == preview.get("input_version"):
        motions += [item for item in partial.get("motions") or [] if isinstance(item, dict)]
    return motions


def best_motion(motions: list[dict]) -> dict | None:
    """The playable motion to show: complete recordings before previews, then most frames."""
    playable = [item for item in motions if _frames(item) >= PLAYABLE_FRAMES]
    return max(playable, key=lambda item: (item.get("status") != "UNVERIFIED_PREFIX", _frames(item))) if playable else None


def _revisions(project: Path) -> list[Path]:
    return sorted((path for path in project.glob("runs/r[0-9]*") if path.is_dir()), reverse=True)


def _sample_count(poses: Path) -> int:
    try:
        with poses.open(encoding="utf-8") as stream:
            return sum(1 for line in stream if line.strip())
    except OSError:
        return 0


def _compute(argv, workspace: Path, project: Path, inputs, outputs, timeout: float):
    from ..runtime.bounded_subprocess import BoundedSubprocessError, run_bounded_subprocess
    from .execution import run_compute
    env = dict(os.environ)
    env.pop("SIMREADY_CONSTRUCTION_PROJECT", None)
    try:
        return run_compute(argv, workspace=workspace, project=project, inputs=inputs, outputs=outputs,
                           timeout_seconds=timeout, cwd=workspace, env=env, runner=run_bounded_subprocess)
    except BoundedSubprocessError as exc:
        return exc


def _replay(workspace: Path, project: Path, prepared: Path, poses: Path, trial: str, status: str,
            output: Path) -> dict | None:
    """Export recorded poses as a dashboard replay; None when that fails."""
    request = output / "replay.request.json"
    write_json_atomic(request, {"prepared": str(prepared), "poses": str(poses), "trial": trial,
                                "status": status, "output": str(output)})
    result = _compute([sys.executable, "-m", TEMPLATE_TASKS, "replay", str(request)], workspace, project,
                      [prepared.parent, poses, request], [output], REPLAY_TIMEOUT_SECONDS)
    assets = _read(output / "replay-assets.json")
    if getattr(result, "returncode", 1) or not assets.get("replay_file"):
        return None
    samples = _read(output / "native-trace.json").get("samples") or []
    return {"motion_model_path": assets["model_file"], "scene_path": assets["scene_file"],
            "replay_path": assets["replay_file"], "motion_trace": str(output / "native-trace.json"),
            "motion_frame_count": min(720, int(assets.get("sample_count") or 0)),
            "recorded_duration_s": round(float(samples[-1].get("time_s") or 0.0), 3) if samples else 0.0,
            "status": "UNVERIFIED_PREFIX"}


def _add_partial(run: Path, motions: list[dict]) -> None:
    preview = _read(run / "slim_evidence" / "preview.json")
    path = run / "slim_evidence" / "partial-motion-preview.json"
    partial = _read(path)
    kept = partial.get("motions") if partial.get("input_version") == preview.get("input_version") else []
    write_json_atomic(path, {"schema_version": PARTIAL_SCHEMA, "input_version": preview.get("input_version"),
                             "motions": [*(kept or []), *motions]})


def _salvage(workspace: Path, project: Path, run: Path) -> list[dict]:
    """Replays for recorded test poses that were never exported (e.g. the run ended first)."""
    if not (run / "slim_evidence" / "preview.json").is_file():
        return []
    exported = {item.get("load_case_id") for item in revision_motions(run)}
    added = []
    for attempt in sorted(run.glob("template_runs/a*"), reverse=True):
        prepared = attempt / "prepared" / "prepared.json"
        if not prepared.is_file():
            continue
        for poses in sorted(attempt.glob("tests/*/*/body_poses.jsonl")):
            trial = poses.parent.name
            if trial in exported or _sample_count(poses) < PLAYABLE_FRAMES:
                continue
            outcome = _read(attempt / "tests" / (trial + ".outcome.json"))
            motion = _replay(workspace, project, prepared, poses, trial,
                             str(outcome.get("execution_status") or "incomplete"), poses.parent)
            if motion:
                added.append({"id": trial, "label": trial, "load_case_id": trial, **motion,
                              "preview_kind": "recorded_test"})
                exported.add(trial)
        if added:
            break
    if added:
        _add_partial(run, added)
    return added


def _settle(workspace: Path, project: Path, run: Path, prepared: Path, chrono_python: str) -> tuple[dict | None, str]:
    """Let the prepared design settle under gravity, without operations, and replay it."""
    # Inside the attempt that prepared the design: the only place sandboxed runs may write.
    output = prepared.parent.parent / "motion_preview" / "gravity_settle"
    shutil.rmtree(output, ignore_errors=True)  # only this module writes here; the native run needs a new folder
    output.mkdir(parents=True)
    request = output / "settle.request.json"
    write_json_atomic(request, {"prepared": str(prepared), "output": str(output / "run"),
                                "duration_s": SETTLE_SECONDS, "step_s": SETTLE_STEP_SECONDS})
    # A settle that diverges or times out still leaves the poses recorded until then.
    result = _compute([str(chrono_python), "-X", "utf8", "-m", __name__, "settle", str(request)], workspace, project,
                      [prepared.parent, request], [output], SETTLE_TIMEOUT_SECONDS)
    poses = output / "run" / "body_poses.jsonl"
    if _sample_count(poses) < PLAYABLE_FRAMES:
        stderr = getattr(result, "stderr", b"") or b""
        detail = (stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else str(stderr))[-600:] or str(result)[-600:]
        return None, "the gravity settle recorded no motion: " + detail.strip()
    motion = _replay(workspace, project, prepared, poses, "gravity_settle", "preview", output)
    if not motion:
        return None, "the gravity settle motion could not be exported"
    motion = {"id": "gravity_settle", "label": "gravity_settle", "load_case_id": "gravity_settle", **motion,
              "preview_kind": "gravity_settle"}
    _add_partial(run, [motion])
    return motion, ""


def ensure_final_motion(workspace, project, *, chrono_python) -> dict:
    """Make sure the project has at least one playable motion; record what was done."""
    workspace, project = Path(workspace).resolve(), Path(project).resolve()
    runs = _revisions(project)
    for run in runs:
        motion = best_motion(revision_motions(run))
        if motion:
            report = {"status": "present", "revision": run.name, "motion": motion.get("id")}
            break
    else:
        report = {"status": "unavailable", "reason": "no recorded poses and no prepared design"}
        for run in runs:
            added = _salvage(workspace, project, run)
            if added:
                report = {"status": "salvaged", "revision": run.name, "motions": [item["id"] for item in added]}
                break
        else:
            prepared = next(((run, path) for run in runs if (run / "slim_evidence" / "preview.json").is_file()
                             for path in sorted(run.glob("template_runs/a*/prepared/prepared.json"), reverse=True)), None)
            if prepared:
                motion, reason = _settle(workspace, project, prepared[0], prepared[1], chrono_python)
                report = ({"status": "settled", "revision": prepared[0].name, "motion": motion["id"]} if motion
                          else {"status": "unavailable", "reason": reason, "revision": prepared[0].name})
    write_json_atomic(project / "final-motion.json", {"schema_version": "slim_final_motion_v1", **report})
    return report


def _settle_main(request_path: Path) -> None:
    from benchmark_templates.cad_native import build_native
    from benchmark_templates.native import run
    request = json.loads(request_path.read_text(encoding="utf-8"))
    model = build_native(request["prepared"])
    run(model, [], duration_s=float(request["duration_s"]), step_s=float(request["step_s"]),
        output=request["output"])


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "settle":
        raise SystemExit("usage: python -m ...slim_harness.motion_preview settle REQUEST.json")
    _settle_main(Path(sys.argv[2]))
