"""Template stages inside Slim's existing build/revise loop; no new physics engine."""
from __future__ import annotations

from contextlib import nullcontext
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

from ..mechanism.io import write_json_atomic
from ..runtime.bounded_subprocess import run_bounded_subprocess, BoundedSubprocessError
from . import modes
from .budget import ActiveBudgetExhausted
from .construction_budget import BudgetExhausted
from .contracts import DeliveryError, load_model_scene
from .template_tasks import DELIVERY_ERROR_EXIT
from .trace_storage import file_digest

BUDGET_ERRORS = (ActiveBudgetExhausted, BudgetExhausted)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def bound_files(paths):
    return {str(Path(p).resolve()): file_digest(Path(p), drop_cache=True) for p in paths}


def retained_files(directory):
    directory = Path(directory)
    return [p for p in directory.rglob("*") if p.is_file()
            and not {"__pycache__", ".git", ".runtime_runs"}.intersection(p.relative_to(directory).parts)]


def implementation_binding():
    """Bind loaded code, independently of the project data workspace."""
    import benchmark_templates
    from .implementation import native_implementation_binding
    directory = Path(__file__).resolve().parent
    files = [directory / name for name in native_implementation_binding()]
    files += [directory / name for name in (
        "template_workflow.py", "template_tasks.py", "audit.py", "geometry.py",
        "physical_parts.py", "static_source_metadata.py", "print_delivery.py", "step_delivery.py")]
    files += list(Path(benchmark_templates.__file__).resolve().parent.glob("*.py"))
    return bound_files(files)


def trial_complete(directory, trial, prepared):
    """A full CSV and complete original-body pose interval must back each verdict."""
    native = read(directory / "summary.json")["trials"]
    if len(native) != 1 or native[0].get("id") != trial["id"]:
        raise ValueError("Native summary does not identify exactly the planned trial")
    result = native[0]
    if result.get("status") != "completed" or result.get("functional_pass") is not True:
        return False
    checks = result.get("checks", [])
    if ([c["id"] for c in checks] != [c["id"] for c in trial["checks"]]
            or not checks or not all(c.get("passed") is True for c in checks)):
        return False
    inner = directory / trial["id"]
    report = read(inner / "report.json")
    if (report.get("status") != "completed"
            or abs(float(report["actual_time_s"]) - trial["duration_s"]) > 1e-9
            or report["steps"] != result["steps"]):
        return False
    rows, first, last, previous = 0, None, None, -1.
    with (inner / "motion.csv").open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            now = float(row["time_s"])
            if not math.isfinite(now) or now <= previous:
                return False
            if abs(float(row["native_time_s"]) - now) > 1e-8:
                return False
            first = now if first is None else first
            previous = last = now
            rows += 1
    if first != 0. or last is None or abs(last - trial["duration_s"]) > 1e-9 or rows != report["steps"] + 1:
        return False
    body_ids = {b["id"] for b in prepared["bodies"]}
    conditional = {r["active_element_id"] for r in (prepared.get("validation_scope") or {}).get("flexible_contact_regions", [])
                   if r.get("active_element_id") is not None}
    first, last, previous = None, None, -1.
    with (inner / "body_poses.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            sample = json.loads(line)
            now = float(sample["time_s"])
            if not math.isfinite(now) or now <= previous or set(sample["bodies"]) != body_ids:
                return False
            states = sample.get("equivalent_elasticity", {}).get("elements", {})
            if any(not isinstance(states.get(key, {}).get("active"), bool) for key in conditional):
                return False
            first = now if first is None else first
            previous = last = now
    receipt = read(directory / "native-model-binding.json")
    return (first == 0. and last is not None and abs(last - trial["duration_s"]) < 1e-9
            and receipt.get("source_model_retained") is True
            and receipt.get("source_callbacks_retained") is True
            and set(receipt.get("body_ids", [])) == body_ids and receipt.get("trial_id") == trial["id"])


def run_revision(flow):
    config = modes.settings(flow.project)
    run = flow.project / "runs" / f"r{flow.state['revision']:02d}"
    source = run / "design/scene.json"
    root = run / "template_runs"
    root.mkdir(exist_ok=True)
    index = max([int(p.name[1:]) for p in root.glob("a[0-9]*") if p.name[1:].isdigit()] + [0]) + 1
    attempt = root / f"a{index:03d}"
    attempt.mkdir()
    started = time.monotonic()
    summaries, motions, program_evidence = [], [], []
    plan, geometry, before = {}, {}, {}
    implementation = implementation_binding()
    prepared_file = attempt / "prepared/prepared.json"
    review_receipt = attempt / "template-review.json"
    plan_file = attempt / "plan.json"

    def persist():
        write_json_atomic(flow.project / "template-progress.json",
                          {"attempt": str(attempt), "execution_mode": config["mode"], "plan": plan, "tests": summaries})
        for row in summaries:
            write_json_atomic(attempt / "tests" / (row["test_id"] + ".outcome.json"), row)

    # Advisory findings about floating parts reach the author with any result but never decide it.
    assembly_warnings = []

    def finish(status, issues, **details):
        persist()
        if assembly_warnings and status != "PASS":
            issues = [*issues, *("Assembly check (advisory): " + warning for warning in assembly_warnings)]
        attempted = {row["test_id"] for row in summaries}
        completed = {row["test_id"] for row in summaries if row.get("execution_status") == "completed"}
        result = {"status": status, "issues": issues, "attempt": str(attempt), "tests": summaries,
                  "completed_tests": summaries, "motions": motions, "geometry": geometry,
                  "pending_tests": [t["id"] for t in plan.get("trials", []) if t["id"] not in completed],
                  "unrun_tests": [t["id"] for t in plan.get("trials", []) if t["id"] not in attempted],
                  "execution_mode": config["mode"], "template_result": True,
                  "complete_instance_pass": status == "PASS", "source_bindings": before,
                  "criteria_sha256": flow.criteria_sha256, "template_plan": str(plan_file),
                  "template_review": str(review_receipt), "prepared_file": str(prepared_file),
                  "execution_implementation": implementation,
                  "revision_execution_seconds": time.monotonic() - started,
                  "validation_scope": geometry.get("validation_scope"),
                  "scope": "Existing family-template native tasks; original source geometry audited at 10 Hz saved poses and exact endpoints; no continuous-time claim"}
        result.update(details)
        if assembly_warnings:
            result["assembly_warnings"] = list(assembly_warnings)
        result["template_evidence_bindings"] = bound_files(
            p for p in [*retained_files(attempt), *program_evidence]
            if p.name not in ("capture.json", "capture-error.json", "publisher.log"))
        write_json_atomic(attempt / "template-result.json", result)
        return result

    def command(operation, request, python, timeout):
        from ..trajectory.capture import HarnessProgramCapture
        path = attempt / (operation + f"-{time.time_ns()}.request.json")
        write_json_atomic(path, request)
        env = dict(os.environ)
        env.pop("SIMREADY_CONSTRUCTION_PROJECT", None)
        prior = (flow.project / "worker_turns").glob("t*_*")
        indices = [int(p.name.split("_", 1)[0][1:]) for p in prior
                   if p.name.split("_", 1)[0][1:].isdigit()]
        turn_dir = flow.project / "worker_turns" / f"t{max(indices, default=0)+1:03d}_harness_{operation}"
        phase = flow.state.get("phase", "template_" + operation)
        argv = [str(python), "-m", __package__ + ".template_tasks", operation, str(path)]
        flow.publish(phase, active_turn={"id": turn_dir.name, "directory": str(turn_dir),
                                         "source": "harness_program"})
        try:
            with HarnessProgramCapture(source.parent, turn_dir) as capture:
                with capture.command(json.dumps(argv, ensure_ascii=False)) as event:
                    from .construction_limits import simulation_activity
                    with flow.budget.active("physical_simulation" if operation == "run" else "template_" + operation), \
                            (simulation_activity(flow.project) if operation == "run" and flow.building else nullcontext()):
                        from .execution import run_compute
                        result = run_compute(
                            argv, cwd=flow.workspace, env=env, timeout_seconds=timeout, max_rss_bytes="auto",
                            workspace=flow.workspace,project=flow.project,inputs=[source.parent,path],outputs=[attempt],
                            experimental_no_timeout=timeout is None,runner=run_bounded_subprocess)
                    stdout = result.stdout.decode("utf-8", errors="replace")
                    stderr = result.stderr.decode("utf-8", errors="replace")
                    event.update(exit_code=result.returncode, stdout=stdout, stderr=stderr,
                                 aggregated_output="\n".join(text for text in (stdout, stderr) if text))
                    path.with_suffix(".stdout.log").write_bytes(result.stdout)
                    path.with_suffix(".stderr.log").write_bytes(result.stderr)
                    (turn_dir / "stdout.log").write_bytes(result.stdout)
                    (turn_dir / "stderr.log").write_bytes(result.stderr)
                    if result.returncode == DELIVERY_ERROR_EXIT:
                        try:
                            delivery = json.loads(stdout.strip().splitlines()[-1])["delivery_error"]
                        except (IndexError, KeyError, TypeError, ValueError):
                            delivery = None
                        if delivery:
                            raise DeliveryError(delivery)
                    if result.returncode:
                        raise RuntimeError(stderr[-6000:])
        finally:
            program_evidence.extend(p for p in (turn_dir / name for name in
                                    ("program.json", "events.jsonl", "stdout.log", "stderr.log")) if p.is_file())
            flow.publish(phase, active_turn={})

    def limit(seconds):
        return flow.budget.limit(seconds)

    try:
        flow.budget.require_available()
        scene = load_model_scene(source)
        from .physical_parts import resolve_physical_part_inventory, check_physical_part_inventory
        from .budget import get_tier
        from .static_source_metadata import grouped_manifest_findings
        from .print_delivery import load_registered_static_evidence
        from .step_delivery import refresh_registered_step_delivery
        task_text = (flow.project / "task.md").read_text(encoding="utf-8")
        inventory = resolve_physical_part_inventory(source, scene, task_text=task_text, project=flow.project)
        write_json_atomic(attempt / "physical-part-inventory.json", inventory)
        # Benchmark/custom families fix the tasks; the tier part cap applies only to free creation.
        inventory_tier = flow.tier or get_tier("high")
        errors = check_physical_part_inventory(inventory, inventory_tier, project=flow.project)
        errors += grouped_manifest_findings(source, read(flow.project / "criteria.json"))
        if errors:
            return finish("NEEDS_REVISION", errors)
        delivery = refresh_registered_step_delivery(flow.project, source, attempt)
        if delivery is not None and delivery.get("status") != "ready":
            return finish("NEEDS_REVISION", delivery.get("issues", []))
        _, static_errors = load_registered_static_evidence(flow.project, source)
        if static_errors:
            return finish("NEEDS_REVISION", static_errors)

        template_source = run / "design/task_template.py" if config["mode"] == "free" else Path(config["template_file"])
        if not template_source.is_file():
            return finish("NEEDS_REVISION", ["Deliver task_template.py for free creation"])
        template_dir = attempt / "template"
        template_files = retained_files(template_source.parent)
        if config["mode"] == "free":
            declared = [relative for relative in scene.get("authoring_sources", [])
                        if Path(relative).suffix.lower() in (".py", ".json")]
            dependency_file = source.parent / "task_template_dependencies.json"
            if dependency_file.is_file():
                dependencies = read(dependency_file)
                if not isinstance(dependencies, list) or any(not isinstance(item, str) for item in dependencies):
                    return finish("NEEDS_REVISION", ["task_template_dependencies.json must contain a list of relative filename strings"])
                declared = [*declared, *dependencies]
            template_files = [template_source, *[(source.parent / relative).resolve() for relative in declared]]
        for path in template_files:
            destination = template_dir / path.relative_to(template_source.parent)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        template_copy = template_dir / template_source.name
        binding_dir = attempt / "reviewer_binding"
        binding_dir.mkdir()
        binding = binding_dir / "binding.py"
        adjustment_file = source.parent / "template_adjustment.json"
        adjustments = read(adjustment_file) if adjustment_file.is_file() else {}
        if adjustments and (not isinstance(adjustments.get("parameters"), dict) or not adjustments.get("reason")):
            return finish("NEEDS_REVISION", ["template_adjustment.json needs parameters and reason"])
        source_files = [*retained_files(source.parent), flow.project / "task.md", flow.project / "criteria.json",
                        flow.project / "mode.json", *retained_files(template_source.parent)]
        source_files += [(source.parent / b["source"]).resolve() for b in scene["bodies"]]
        before = bound_files(source_files)
        if before[str((flow.project / "criteria.json").resolve())] != flow.criteria_sha256:
            return finish("HARNESS_ERROR", ["Criteria changed before template execution"])
        flow.publish("geometry")
        command("prepare", {"scene": str(source), "output": str(prepared_file.parent)}, sys.executable, limit(1800))
        prepared = read(prepared_file)
        scope_bindings = (prepared.get("validation_scope") or {}).get("source_bindings", {})
        if scope_bindings:
            if bound_files(scope_bindings) != scope_bindings:
                return finish("HARNESS_ERROR", ["Equivalent model source evidence changed during preparation"])
            before.update(scope_bindings)
            source_files.extend(Path(path) for path in scope_bindings)
        geometry = read(prepared_file.parent / "geometry/geometry.json")
        if prepared.get("validation_scope"):
            geometry["validation_scope"] = prepared["validation_scope"]
        prepared_sources = retained_files(prepared_file.parent)
        source_files.extend(prepared_sources)
        before.update(bound_files(prepared_sources))
        # The shared capture queues dashboard publication outside native execution.
        from ..trajectory.capture import publish_attempt
        capture_result = {"source": str(source), "command": "template_prepare", "status": "GEOMETRY_READY"}
        publish_attempt(flow.workspace, prepared_file.parent, capture_result, project=flow.project,
                        destination=source, stage="geometry_ready")
        publish_attempt(flow.workspace, prepared_file.parent, capture_result, project=flow.project,
                        destination=source, stage="finished")
        flow._preview(run, geometry, [])
        audit_file = attempt / "initial-geometry-audit.json"
        command("audit", {"prepared": str(prepared_file), "output": str(audit_file)}, sys.executable, limit(1800))
        if not read(audit_file).get("passed"):
            return finish("NEEDS_REVISION", read(audit_file).get("violations", []))
        from .floating_parts import advisory_report
        prepared_model = read(prepared_file)
        assembly = advisory_report(prepared_model["bodies"], prepared_model.get("joints") or [],
                                   attempt / "floating-parts.json")
        assembly_warnings.extend(finding["message"] for finding in (assembly or {}).get("findings", []))
        request = {"template": str(template_copy), "binding": str(binding), "plan": str(plan_file),
                   "requested_parameters": adjustments.get("parameters", {})}
        prompt = (
            f"You are the independent Harness template reviewer and instance filler. Read {flow.workspace / 'benchmark_templates/AUTHORING_API.md'}. "
            + ("Write all newly authored explanations, parameter provenance, code comments and reports in English. "
               if config.get("language") == "en" else "")
            + f"Task: {task_text}\nSource design: {source.parent}\nTemplate: {template_copy}\n"
            f"Prepared source model (read-only): {prepared_file}\n"
            f"Write {binding}, exporting PROTOCOL and build_model(source_dir, step_s, protocol), "
            "or build_model_for_trial(source_dir, step_s, protocol, trial_id). "
            "Use benchmark_templates.cad_native.build_native on the supplied prepared.json. Keep all original "
            "bodies, source callbacks before_step/after_step/commit_inputs and the prepared model unchanged. "
            "Existing observer wrappers may wrap source callbacks, retaining and calling the original hook. "
            "Map observations through real signals and operations, never expected outputs or substitute mechanics. "
            "Keep binding dependencies in the binding directory. "
            + ("Review the newly authored template for prompt coverage, operations and checks; report concrete template defects. "
               if config["mode"] == "free" else "The selected task template is fixed; do not rewrite it. ")
            + "The constructor owns difficulty and parameter decisions. Implement requested parameter values; "
            "record disagreement as advice, not a veto. Technical binding errors should be repaired in the binding. "
            f"Requested decisions: {json.dumps(adjustments, ensure_ascii=False)}\n"
            "Write parameter_sources.json as an object keyed by each used PROTOCOL parameter, with "
            "value (the actual resolved value) and source (source file/geometry/task/constructor decision). "
            "Do not edit the source, template, task, Harness, solver or results. Return ready after filling."
        )
        prior_plans = sorted((flow.project / "runs").glob("r*/template_runs/a*/plan.json"))
        if prior_plans:
            prior = read(prior_plans[-1])
            prompt += ("\nPrior reviewed protocol: " + json.dumps(prior.get("protocol", {}), ensure_ascii=False)
                       + ". Preserve prior difficulty/protocol choices unless the constructor explicitly changes them. "
                       "Rebind source-dependent quantities to the current design and document their provenance.")
        replies = []
        # One immediate correction opportunity for a malformed/missing binding,
        # without sending an implementation error back as a mechanical redesign.
        for filling in range(2):
            flow.publish("template_review", active_test=None)
            response = flow.turn("template_review_fill", lambda directory: flow.reviewer.run_turn(
                binding_dir, prompt, directory, limit(2400), fresh=True))
            replies.append(response)
            write_json_atomic(review_receipt, {"responses": replies, "source_bindings": before,
                                              "requested_parameters": request["requested_parameters"],
                                              "binding_file": str(binding), "ready": False})
            if response.get("status") == "provider_error":
                return finish("provider_error", [response], **{
                    key: response[key] for key in ("provider_error", "summary", "recoverable", "needs_external_action")
                    if key in response})
            if bound_files(source_files) != before or implementation_binding() != implementation:
                return finish("HARNESS_ERROR", ["Source/template or Harness implementation changed during independent filling"])
            if response.get("status") != "ready" or not binding.is_file():
                if filling == 0:
                    prompt += "\nThe last turn did not deliver a ready binding.py. Correct the binding and return ready."
                    continue
                technical = config["mode"] != "free" or response.get("status") == "ready"
                return finish("EVIDENCE_REQUIRED" if technical else "NEEDS_REVISION", [response],
                              template_binding_error=technical)
            try:
                command("plan", {**request, "output": str(plan_file)}, flow.chrono_python, limit(900))
                plan = read(plan_file)
                if not plan["trials"]:
                    return finish("NEEDS_REVISION", ["Template plan declares no trials"])
                if config["mode"] == "free":
                    # A free-creation template is author-declared, so its trial count follows the selected test limit.
                    maximum = flow.construction.snapshot()["config"]["max_tests"] if flow.construction else (flow.tier.max_tasks_per_design if flow.tier else 5)
                    if len(plan["trials"]) > maximum:
                        return finish("NEEDS_REVISION", ["Template exceeds the selected test limit"])
                command("precheck", {**request, "prepared": str(prepared_file),
                                      "output": str(attempt / "binding_checks" / f"fill{filling+1:02d}"),
                                      "step_s": min(.001, float(getattr(flow, "physics_max_step_seconds", .001)))},
                        flow.chrono_python, limit(1800))
                break
            except BUDGET_ERRORS:
                raise
            except Exception as exc:
                if filling:
                    return finish("EVIDENCE_REQUIRED", [f"Reviewer binding failed technical precheck: {exc}"], template_binding_error=True)
                prompt += f"\nThe binding's technical validation failed. Correct binding/provenance only: {exc}"
        write_json_atomic(review_receipt, {"responses": replies, "source_bindings": before,
                                          "requested_parameters": request["requested_parameters"],
                                          "binding_file": str(binding), "ready": True})
        fill_bindings = bound_files([*retained_files(binding_dir), *retained_files(template_dir), plan_file, review_receipt])
        persist()
        for trial in plan["trials"]:
            flow.budget.require_available()
            flow.publish("physics", active_test=trial["id"])
            output = attempt / "tests" / trial["id"]
            configured = flow.construction.snapshot()["config"]["simulation_timeout_seconds"] if flow.construction else flow.physics_timeout_seconds
            if flow.building:
                # Construction simulations are not charged and extend past their configured limit.
                from .construction_limits import simulation_time_steps
                timeout = simulation_time_steps(configured)[-1]
            else:
                timeout = limit(configured)
            row = {"test_id": trial["id"], "execution_started": True, "physical_pass": False,
                   "execution_status": "running", "assertions": [],
                   "native_summary_file": str(output / "summary.json"),
                   "native_csv_file": str(output / trial["id"] / "motion.csv"),
                   "native_poses_file": str(output / trial["id"] / "body_poses.jsonl"),
                   "geometry_audit": {"passed": False, "status": "not_run"}}
            summaries.append(row)
            persist()
            started = time.monotonic()
            try:
                command("run", {**request, "prepared": str(prepared_file), "output": str(output),
                                "trial": trial["id"], "step_s": min(.001, float(getattr(flow, "physics_max_step_seconds", .001)))}, flow.chrono_python, timeout)
                native = read(output / "summary.json")["trials"][0]
                row.update(execution_status=native["status"], assertions=native.get("checks", []),
                           physical_pass=trial_complete(output, trial, prepared))
                if not row["physical_pass"]:
                    row["failure"] = native
            except BUDGET_ERRORS as exc:
                row.update(execution_status="BUDGET_EXHAUSTED", failure={"issues": [str(exc)]})
                raise
            except BoundedSubprocessError as exc:
                row.update(execution_status="SIMULATION_TIMEOUT" if exc.code == "BOUNDED_SUBPROCESS_TIMEOUT" else "RESOURCE_ERROR",
                           failure={"issues": [str(exc)]})
                flow.budget.require_available()
            except Exception as exc:
                row.update(execution_status="HARNESS_ERROR", failure={"issues": [str(exc)]})
            finally:
                if flow.building and timeout is not None:
                    from .construction_limits import simulation_extension
                    elapsed = time.monotonic() - started
                    row.update(simulation_wall_seconds=elapsed, **simulation_extension(configured, elapsed))
                persist()
        # All trials get their first opportunity before optional replay and source-motion audit.
        for row in summaries:
            output = attempt / "tests" / row["test_id"] / row["test_id"]
            if not Path(row["native_poses_file"]).is_file():
                continue
            try:
                command("replay", {"prepared": str(prepared_file), "poses": row["native_poses_file"],
                                  "trial": row["test_id"], "status": row["execution_status"],
                                  "output": str(output)}, sys.executable, limit(900))
                assets = read(output / "replay-assets.json")
                row["trace_file"] = str(output / "native-trace.json")
                motion = {"id": row["test_id"], "label": row["test_id"], "load_case_id": row["test_id"],
                          "motion_model_path": assets["model_file"], "scene_path": assets["scene_file"],
                          "replay_path": assets["replay_file"], "motion_trace": row["trace_file"],
                          "motion_frame_count": min(720, assets["sample_count"]),
                          "replay_version": file_digest(Path(row["trace_file"]), drop_cache=True)}
                motions.append(motion)
                flow._preview(run, geometry, [], motions)
            except BUDGET_ERRORS:
                raise
            except Exception as exc:
                row["replay_error"] = str(exc)
            if not row["physical_pass"]:
                row["geometry_audit"] = {"passed": False, "status": "not_completed_for_rejected_test"}
                persist()
                continue
            flow.publish("geometry_audit", active_test=row["test_id"])
            try:
                audit_file = output / "geometry-audit.json"
                command("audit", {"prepared": str(prepared_file), "poses": row["native_poses_file"],
                                  "output": str(audit_file)}, sys.executable, limit(1800))
                row["geometry_audit"] = read(audit_file)
            except BUDGET_ERRORS:
                raise
            except Exception as exc:
                row["geometry_audit"] = {"passed": False, "status": "audit_error", "issues": [str(exc)]}
            persist()
        _, static_errors = load_registered_static_evidence(flow.project, source)
        if static_errors:
            return finish("NEEDS_REVISION", static_errors)
        if (bound_files(source_files) != before or bound_files(fill_bindings) != fill_bindings
                or implementation_binding() != implementation):
            return finish("HARNESS_ERROR", ["Source, filled template or implementation changed during validation"])
        technical = [row for row in summaries if row.get("execution_status") in
                     {"binding_or_execution_error", "assessment_error", "HARNESS_ERROR"}]
        if technical:
            return finish("EVIDENCE_REQUIRED", ["Template binding or assessment requires technical repair", *technical],
                          template_binding_error=True)
        success = len(summaries) == len(plan["trials"]) and all(
            row["physical_pass"] and row["geometry_audit"].get("passed") is True for row in summaries)
        if flow.state.get("first_pass_handoff"):
            return finish("NEEDS_REVISION", ["First test coverage retained; author completion is still required"])
        return finish("PASS" if success else "NEEDS_REVISION",
                      [] if success else ["Template tasks or original-source motion checks did not pass"])
    except BUDGET_ERRORS as exc:
        return finish("BUDGET_EXHAUSTED", [str(exc)])
    except DeliveryError as exc:
        # Same rule as the direct-scene path: the author fixes its delivery; only a capability gap is the Harness's.
        return finish("HARNESS_ERROR" if str(exc).startswith("CAPABILITY_GAP:") else "NEEDS_REVISION", [str(exc)])
    except Exception as exc:
        return finish("HARNESS_ERROR", [f"{type(exc).__name__}: {exc}"])


def publication_valid(value, run):
    """Recheck the current source, reviewed fill, implementation and complete trials."""
    try:
        run = Path(run).resolve()
        project = run.parent.parent
        from ..snapshot_paths import workspace_for, resolve, binding_paths, implementation_matches
        archive = workspace_for(run)
        def local_path(path):
            return resolve(archive, path) if archive else Path(path).resolve()
        config = modes.settings(project)
        if (not config or value.get("execution_mode") != config["mode"]
                or value.get("status") != "PASS" or value.get("template_result") is not True
                or value.get("pending_tests") or value.get("unrun_tests")):
            return False
        attempt = local_path(value["attempt"])
        if attempt.parent != run / "template_runs":
            return False
        for group in ("source_bindings", "template_evidence_bindings"):
            bindings = value.get(group)
            if archive and bindings:
                bindings = binding_paths(archive, bindings)
            if not bindings or bound_files(bindings) != bindings:
                return False
        implementation = value.get("execution_implementation")
        current = implementation_binding()
        if not implementation:
            return False
        if archive:
            # The receiving implementation must still match every saved digest;
            # only its workspace prefix is allowed to differ.
            if not implementation_matches(archive, implementation, current):
                return False
        elif bound_files(implementation) != implementation or current != implementation:
            return False
        source = run / "design/scene.json"
        if source not in {local_path(path) for path in value["source_bindings"]}:
            return False
        if file_digest(project / "criteria.json", drop_cache=True) != value["criteria_sha256"]:
            return False
        review = read(local_path(value["template_review"]))
        if review.get("ready") is not True or not review.get("responses") or review["responses"][-1].get("status") != "ready":
            return False
        if not review.get("source_bindings") or any(value["source_bindings"].get(path) != digest
                                                    for path, digest in review["source_bindings"].items()):
            return False
        plan = read(local_path(value["template_plan"]))
        rows = value.get("tests", [])
        if [row["test_id"] for row in rows] != [trial["id"] for trial in plan["trials"]] or not rows:
            return False
        prepared = read(local_path(value["prepared_file"]))
        if not read(attempt / "initial-geometry-audit.json").get("passed"):
            return False
        for row, trial in zip(rows, plan["trials"]):
            directory = attempt / "tests" / row["test_id"]
            native = read(directory / "summary.json")["trials"][0]
            audit = read(directory / row["test_id"] / "geometry-audit.json")
            if (read(attempt / "tests" / (row["test_id"] + ".outcome.json")) != row
                    or row.get("execution_status") != "completed" or row.get("physical_pass") is not True
                    or row.get("assertions") != native.get("checks") or row.get("geometry_audit") != audit
                    or audit.get("passed") is not True or not trial_complete(directory, trial, prepared)):
                return False
        from .print_delivery import load_registered_static_evidence
        _, errors = load_registered_static_evidence(project, source)
        return not errors
    except (OSError, ValueError, KeyError, TypeError):
        return False
