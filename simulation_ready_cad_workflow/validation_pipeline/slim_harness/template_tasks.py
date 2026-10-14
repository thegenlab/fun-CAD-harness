"""Process boundaries for existing template execution and original-source evidence."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

from ..mechanism.io import write_json_atomic

# Exit status for a DeliveryError: the author's delivery, not the Harness, must change.
DELIVERY_ERROR_EXIT = 3


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def saved_samples(path):
    """Retain source state; only translate the native pose field names."""
    samples = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            bodies = row.pop("bodies")
            samples.append({**row, "body_poses": {
                key: {"position_abs_m": pose["position_m"],
                      "rotation_abs_quat_wxyz": pose["rotation_wxyz"]}
                for key, pose in bodies.items()}})
    if not samples:
        raise ValueError("Native execution supplied no saved original-body poses")
    return samples


def resolved_plan(template, protocol):
    return {"family_id": template.family_id, "variant": template.variant,
            "source_task_ids": list(template.source_task_ids), "protocol": protocol,
            "required_signals": list(template.required_signals),
            "trials": [asdict(trial) for trial in template.trials]}


def requested_parameter_value(value):
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def constructor_parameters_match(protocol, requested):
    """Compare used protocol values to constructor decisions.

    ``template_adjustment.json`` may store a scalar or a ``{value, reason}``
    object. Unused requested names are constructor notes, not binding errors.
    """
    if not requested:
        return True
    return all(
        key not in requested or used == requested_parameter_value(requested[key])
        for key, used in protocol.items())


def retains_callback(callback, original):
    """Accept existing observer/source-scope wrappers that retain the source hook.

    The public observer helpers close over their previous hook; decorated and
    partial wrappers can expose the same chain explicitly. This checks retained
    callback references, not a substitute for reviewing the observer's behavior.
    """
    if original is None:
        return True
    pending, seen = [callback], set()
    while pending:
        current = pending.pop()
        if current is original:
            return True
        if not callable(current) or id(current) in seen:
            continue
        seen.add(id(current))
        pending.extend((getattr(current, "__wrapped__", None), getattr(current, "func", None)))
        pending.extend(getattr(current, "__defaults__", None) or ())
        pending.extend((getattr(current, "__kwdefaults__", None) or {}).values())
        for cell in getattr(current, "__closure__", None) or ():
            try:
                pending.append(cell.cell_contents)
            except ValueError:
                pass  # An empty closure cell cannot preserve a source callback.
    return False


def checked_binding(binding, request, runtime, created):
    """Check the reviewer retained the complete source model, before any stepping."""
    prepared = read(request["prepared"])

    def build(source_dir, step_s, protocol, trial_id):
        trial = SimpleNamespace(id=trial_id)
        model = runtime.build_trial_model(binding, source_dir, step_s, protocol, trial)
        matching = [row for row in created if row["system"] is model.system]
        if len(matching) != 1:
            raise ValueError("Binding must use the supplied cad_native.build_native source model")
        original = matching[0]
        objects = getattr(model, "objects", {})
        bodies = objects.get("bodies", {})
        if (objects.get("prepared") != prepared or set(bodies) != {b["id"] for b in prepared["bodies"]}
                or set(bodies) != set(original["bodies"])
                or any(bodies[key] is not body for key, body in original["bodies"].items())):
            raise ValueError("Binding changed the original source body inventory or prepared physics")
        for hook in ("before_step", "after_step", "commit_inputs"):
            if not retains_callback(getattr(model, hook), original[hook]):
                raise ValueError("Binding must retain the original source callback: " + hook)
        receipt = {"source_model_retained": True, "body_ids": sorted(bodies),
                   "prepared": str(Path(request["prepared"]).resolve()),
                   "source_callbacks_retained": True, "trial_id": trial_id,
                   "callback_check": "Original hook identity or retained observer/source-scope wrapper chain"}
        write_json_atomic(Path(request["output"]) / "native-model-binding.json", receipt)
        return model

    return SimpleNamespace(build_model_for_trial=build)


def precheck_binding(template, binding, protocol, request, runtime, created):
    """Load each actual source instance and read its bindings without stepping."""
    rows = []
    for trial in template.trials:
        created.clear()
        checked = checked_binding(binding, {**request, "output": str(Path(request["output"]) / trial.id)}, runtime, created)
        model = runtime.build_trial_model(checked, request["prepared"], request["step_s"], protocol, trial)
        required = set(template.required_signals) | set(trial.required_signals)
        required.update(name for check in trial.checks for name in check.signals())
        required -= {"time_s", "native_time_s"}
        missing = required - set(model.signals)
        if missing:
            raise ValueError("Missing required observables: " + str(sorted(missing)))
        missing_ports = {action.port for action in trial.actions} - set(model.actuators)
        if missing_ports:
            raise ValueError("Missing actuators: " + str(sorted(missing_ports)))
        for action in trial.actions:
            action.operation().validate()
        if abs(float(model.system.GetChTime())) > 1e-12:
            raise ValueError("Expected a freshly constructed system at t=0")
        for name, signal in model.signals.items():
            if not math.isfinite(float(signal.read())):
                raise ValueError("Non-finite initial observable: " + name)
        rows.append({"trial_id": trial.id, "status": "bindings_ready", "stepped": False})
        del model
    write_json_atomic(Path(request["output"]) / "precheck.json", {"status": "ready", "trials": rows, "stepped": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("prepare", "plan", "precheck", "run", "audit", "replay"))
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request = read(args.request)
    if args.operation == "prepare":
        from benchmark_templates.cad_native import prepare
        prepare(request["scene"], request["output"])
        return
    if args.operation in ("audit", "replay"):
        prepared = read(request["prepared"])
        if args.operation == "replay":
            from .evidence import export_replay
            geometry = prepared["cad_loader"]["source_geometry"]
            geometry = dict(geometry)
            samples = saved_samples(request["poses"])
            trace = {"test_id": request["trial"], "samples": samples,
                     "status": request["status"], "source_kind": "original_geometry",
                     "scope": "Recorded native poses only; force and deformation fields are unavailable"}
            if prepared.get("validation_scope"):
                trace["validation_scope"] = geometry["validation_scope"] = prepared["validation_scope"]
            directory = Path(request["output"])
            write_json_atomic(directory / "native-trace.json", trace)
            assets = export_replay(geometry, trace, directory / "replay")
            write_json_atomic(directory / "replay-assets.json", {**assets, "sample_count": len(samples)})
            return
        from .audit import audit_motion, audit_saved_motion
        if request.get("poses"):
            audit_saved_motion(prepared, saved_samples(request["poses"]), Path(request["output"]))
        else:
            poses = {b["id"]: {"position_abs_m": b["position_m"],
                                "rotation_abs_quat_wxyz": b["rotation_wxyz"]}
                     for b in prepared["bodies"]}
            audit_motion({"bodies": prepared["bodies"]},
                         [{"time_s": 0., "body_poses": poses}], Path(request["output"]))
        return

    from benchmark_templates import runtime
    created = []
    if args.operation in ("run", "precheck"):
        # Observe the existing source loader; never replace or advance its solver.
        from benchmark_templates import cad_native
        source_build = cad_native.build_native

        def tracked_build(*positional, **keywords):
            model = source_build(*positional, **keywords)
            created.append({"system": model.system, "bodies": dict(model.objects["bodies"]),
                            **{name: getattr(model, name) for name in
                               ("before_step", "after_step", "commit_inputs")}})
            return model

        cad_native.build_native = tracked_build
    template_module = runtime.load_source_module(request["template"])
    binding = runtime.load_source_module(request["binding"])
    values = runtime.ProtocolValues(template_module.PARAMETERS, binding.PROTOCOL)
    template = template_module.build(values).validate()
    protocol = dict(values.used)
    if not constructor_parameters_match(protocol, request.get("requested_parameters") or {}):
        raise ValueError("Reviewer binding did not implement the constructor's parameter decision")
    sources = read(Path(request["binding"]).parent / "parameter_sources.json")
    for key in protocol:
        entry = sources.get(key)
        if not isinstance(entry, dict) or entry.get("value") != protocol[key] or not entry.get("source"):
            raise ValueError("Parameter provenance requires value and source for " + key)
    plan = resolved_plan(template, protocol)
    if args.operation == "plan":
        write_json_atomic(Path(request["output"]), plan)
        return
    if json.loads(json.dumps(plan)) != read(request["plan"]):
        raise ValueError("Resolved template changed after independent filling")
    if args.operation == "precheck":
        precheck_binding(template, binding, protocol, request, runtime, created)
        return
    runtime.execute(template, checked_binding(binding, request, runtime, created), protocol,
                    source_dir=request["prepared"], output=request["output"],
                    step_s=request["step_s"], trial_id=request["trial"])


if __name__ == "__main__":
    from .contracts import DeliveryError
    try:
        main()
    except DeliveryError as exc:
        print(json.dumps({"delivery_error": str(exc)}))
        sys.exit(DELIVERY_ERROR_EXIT)
