"""Read the slim program outcome without projecting a legacy certificate."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

CODE_ACCEPTANCE_POLICY = "slim_code_gates_v1"
LEGACY_ACCEPTANCE_POLICY = "slim_independent_model_review_legacy"


def code_gate_failures(scene, tests):
    """Require every declared test and its existing physical/geometry gates."""
    declared=[row["id"] for row in scene.get("tests",[])]
    observed=[row.get("test_id") for row in tests]
    failures=[]
    if (not declared or len(observed)!=len(set(observed)) or len(declared)!=len(set(declared))
            or set(observed)!=set(declared)):
        failures.append("Complete, unique results are required for every declared scene test")
    for row in tests:
        audit=row.get("geometry_audit",{})
        assertions=row.get("assertions") or []
        if (row.get("physical_pass") is not True or row.get("failure") is not None
                or row.get("numerical_failure_unresolved") or audit.get("passed") is not True
                or audit.get("violations") or not assertions
                or any(item.get("passed") is not True for item in assertions)):
            failures.append("Physical, assertion or original-geometry gate did not pass: "+str(row.get("test_id")))
    return failures


def read_result(run: Path) -> dict:
    path=Path(run)/"result.json"
    if not path.is_file(): return {}
    try:
        from ..snapshot_paths import resolve, workspace_for
        workspace = workspace_for(run) or Path.cwd()
        def bound_path(value):
            return resolve(workspace, value)
        value=json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version")!="slim_harness_result_v1" or f"r{int(value['revision']):02d}"!=run.name:
            return {}
        value["eligible_pass"]=False
        if value.get("status")!="PASS": return value
        if value.get("template_result"):
            from .template_workflow import publication_valid
            value["eligible_pass"]=publication_valid(value,run)
            if not value["eligible_pass"]:value["status"]="INCOMPLETE_EVIDENCE"
            return value
        bindings=value.get("source_bindings") or {}
        if not bindings: return {**value,"status":"STALE_EVIDENCE"}
        scene_file=run/"design"/"scene.json"
        scene=json.loads(scene_file.read_text(encoding="utf-8"))
        project=run.parent.parent
        experiment=None
        if (project/"workflow.json").is_file():
            from .frozen_experiment import load_experiment, bind_scene
            try:
                experiment=load_experiment(project)
                if experiment is not None and bind_scene(scene,experiment)!=scene:
                    return {**value,"status":"INCOMPLETE_EVIDENCE"}
            except ValueError:
                return {**value,"status":"STALE_EVIDENCE"}
        elif (project/"frozen_experiment.json").is_file():
            return {**value,"status":"STALE_EVIDENCE"}
        from .validation_scope import validate_result_scope, ValidationScopeError
        try:
            scope=validate_result_scope(value,scene)
        except ValidationScopeError:
            return {**value,"status":"INCOMPLETE_EVIDENCE"}
        modeling_policy_file=run.parent.parent/"modeling_policy.json"
        if modeling_policy_file.is_file():
            policy=json.loads(modeling_policy_file.read_text(encoding="utf-8"))
            if (policy.get("selected_mode")=="equivalent_elastic"
                    and (scope is None or scope["mode"]!="equivalent_elastic")):
                return {**value,"status":"INCOMPLETE_EVIDENCE"}
            if {bound_path(k):v for k,v in bindings.items()}.get(modeling_policy_file.resolve())!=hashlib.sha256(modeling_policy_file.read_bytes()).hexdigest():
                return {**value,"status":"STALE_EVIDENCE"}
        from .measurement_registry import load_registered_measurements, validate_trace_measurements, MeasurementCapabilityError
        try:
            measurements=load_registered_measurements(run.parent.parent,scene_file)
        except MeasurementCapabilityError:
            return {**value,"status":"INCOMPLETE_EVIDENCE"}
        allowed_sources={scene_file.resolve(),*[(scene_file.parent/body["source"]).resolve() for body in scene["bodies"]]}
        if experiment is not None:allowed_sources.add((project/"frozen_experiment.json").resolve())
        if modeling_policy_file.is_file():
            allowed_sources.add(modeling_policy_file.resolve())
        if scope:
            allowed_sources.update(bound_path(path) for path in scope["source_bindings"])
            if any(bindings.get(path)!=expected for path,expected in scope["source_bindings"].items()):
                return {**value,"status":"STALE_EVIDENCE"}
            for body in scene["bodies"]:
                bound=scope["body_inventory"][body["id"]]
                if (scene_file.parent/body["source"]).resolve()!=bound_path(bound["source_file"]):
                    return {**value,"status":"STALE_EVIDENCE"}
            value["scope"]=scope["display_name"]+"; "+scope["dynamic_geometry"]
            if scene.get("spur_gears"):
                from .spur_gears import SPUR_MODEL_SCOPE
                value["scope"] += "; " + SPUR_MODEL_SCOPE
        if measurements:
            allowed_sources.update({(run.parent.parent/"measurement-registration.json").resolve(),
                                    (run.parent.parent/"task.md").resolve()})
            allowed_sources.update((scene_file.parent/relative).resolve()
                for binding in measurements.values() for relative in binding["source_plan_bindings"])
        if {bound_path(path) for path in bindings}!=allowed_sources:
            return {**value,"status":"STALE_EVIDENCE"}
        for source,expected in bindings.items():
            target=bound_path(source)
            if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
                return {**value,"status":"STALE_EVIDENCE"}
        criteria_file=run.parent.parent/"criteria.json"
        if hashlib.sha256(criteria_file.read_bytes()).hexdigest()!=value.get("criteria_sha256"):
            return {**value,"status":"STALE_EVIDENCE"}
        criteria=json.loads(criteria_file.read_text(encoding="utf-8"))
        if experiment is not None and criteria["criteria"]!=experiment["criteria"]:
            return {**value,"status":"STALE_EVIDENCE"}
        if criteria["task_text"]!=(run.parent.parent/"task.md").read_text(encoding="utf-8"):
            return {**value,"status":"STALE_EVIDENCE"}
        tests=value.get("tests") or []
        if not tests or any(not x.get("physical_pass") or not x.get("geometry_audit",{}).get("passed") for x in tests):
            return {**value,"status":"INCOMPLETE_EVIDENCE"}
        try:
            for test in tests:
                validate_trace_measurements(test,measurements.get(test.get("test_id")))
        except MeasurementCapabilityError:
            return {**value,"status":"INCOMPLETE_EVIDENCE","eligible_pass":False}
        policy=value.get("acceptance_policy",LEGACY_ACCEPTANCE_POLICY)
        if policy==CODE_ACCEPTANCE_POLICY:
            from .print_delivery import load_registered_static_evidence
            _,static_errors=load_registered_static_evidence(project,scene_file)
            value["eligible_pass"]=not static_errors and not code_gate_failures(scene,tests)
        elif policy==LEGACY_ACCEPTANCE_POLICY:
            # Preserve the old interpretation; reading history never upgrades
            # an old PASS to the newly authorized code-only policy.
            from .review import review_passes
            from .print_delivery import load_registered_static_evidence
            static_evidence,_=load_registered_static_evidence(project,scene_file)
            value["eligible_pass"]=review_passes(value.get("review",{}),criteria,scene,final=True,
                                                static_evidence=static_evidence,validation_scope=scope)
        else:
            value["eligible_pass"]=False
        value["acceptance_policy"]=policy
        if not value["eligible_pass"]: value["status"]="INCOMPLETE_EVIDENCE"
        return value
    except (OSError,ValueError,KeyError,TypeError):
        return {}
