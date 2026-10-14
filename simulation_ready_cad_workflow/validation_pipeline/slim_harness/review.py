"""Independent, sourced review; criteria and decisions are persisted by code."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from ..mechanism.io import write_json_atomic
from .project_isolation import CODE_ROOT
from .static_delivery import VerifiedStaticEvidence
from .print_delivery import applies, has_static_evidence, is_static_delivery_criterion
from .validation_scope import validate_validation_scope, ValidationScopeError


def obj(properties):
    return {"type":"object","additionalProperties":False,"required":list(properties),"properties":properties}


STRING = {"type":"string"}
STRINGS = {"type":"array","items":STRING}
CRITERIA_SCHEMA = obj({
    "status":{"type":"string","enum":["ready"]},"summary":STRING,
    "criteria":{"type":"array","items":obj({"id":STRING,"kind":{"type":"string","enum":["appearance","function"]},
        "statement":STRING,"source_quote":STRING,"verification":STRING})},
})
REVIEW_SCHEMA = obj({
    "status":{"type":"string","enum":["ready","needs_revision","needs_evidence"]},"summary":STRING,
    "observations":{"type":"array","items":obj({"criterion_id":STRING,
        "verdict":{"type":"string","enum":["met","violated","unobservable"]},
        "observed_fact":STRING,"evidence_paths":STRINGS})},
    "test_coverage":{"type":"array","items":obj({"criterion_id":STRING,"test_ids":STRINGS,
        "appropriate":{"type":"boolean"},"reason":STRING})},
    "static_coverage":{"type":"array","items":obj({"criterion_id":STRING,"receipt_ids":STRINGS})},
    "evidence_requests":{"type":"array","items":obj({"criterion_id":STRING,
        "action":{"type":"string","enum":["additional_view","local_measurement","simulation"]},"target":STRING,"reason":STRING})},
})
EVIDENCE_PLAN_SCHEMA = obj({
    "status":{"type":"string","enum":["ready"]},"summary":STRING,
    "views":{"type":"array","items":obj({"label":STRING,
        "center_m":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3},
        "radius_m":{"type":"number"},"body_ids":STRINGS,"reason":STRING,"evidence_paths":STRINGS,
        "elevation_deg":{"type":"number","minimum":-89,"maximum":89},
        "azimuth_deg":{"type":"number","minimum":-180,"maximum":180}})},
})


def plan_evidence(worker,workspace,scene_path,geometry_file,requests,previous_views,turn_dir):
    return worker.run_turn(workspace,
        "Plan concrete original-geometry close-up views to resolve these existing independent-review "
        "requests. This is evidence collection, not another vote or a design revision. Read only the "
        "current scene, its modeling source files, and prepared geometry metadata to locate the features "
        "in actual WORLD METRES. Do not guess coordinates. Return at most four tightly framed views "
        "with center_m, radius_m, and the body_ids to include (empty means all bodies). The program "
        "will render front, back, side, opposite side, top and perspective views of the original surfaces "
        "in each crop. Include only the relevant source bodies if occlusion hides a feature. Cite the "
        "source files for an explicit elevation_deg and azimuth_deg for the first oblique view; choose "
        "a view that can see the requested surfaces together, rather than repeating fixed camera angles. "
        "source files/parameters that establish each crop. Never modify design files or acceptance "
        "criteria. If previous crops failed, choose concrete new framing or included bodies; do not "
        "repeat the same views.\n"
        f"SCENE: {scene_path}\nPREPARED GEOMETRY: {geometry_file}\n"
        f"REQUESTS: {json.dumps(requests,ensure_ascii=False)}\nPREVIOUS VIEWS: {json.dumps(previous_views,ensure_ascii=False)}",
        turn_dir,1200,readonly=True,fresh=True,response_schema=EVIDENCE_PLAN_SCHEMA)


def make_criteria(worker: Any, workspace: Path, task_path: Path, turn_dir: Path, output: Path,
                  *, task_adjustable: bool = False, tier_config: dict | None = None):
    """Review once before any design exists.

    Default calls retain the original criteria-only behavior. Explicit budgeted
    adjustable calls return reviewed task text and criteria without writing
    either file: the controller must first retain the original and review.
    """
    task = task_path.read_text(encoding="utf-8")
    from .construction import budget_for, construction_requested
    project=task_path.parent;budget=budget_for(project) if construction_requested(project) else None
    construction_note=('Benchmark CONSTRUCTION: group function criteria into at most '+str(budget.snapshot()['config']['max_tasks'])+
        ' core mechanical tasks. Preserve all explicit essential clauses within these groups; appearance criteria may be separate. '
        'Do not invent extra test strictness. Numeric targets explicitly described as suggested/initial construction targets are adjustable, '
        'not mandatory frozen requirements. Group each named task so all core functions remain covered.\n') if budget else ''
    initial_task_review = task_adjustable and tier_config is not None
    schema = CRITERIA_SCHEMA
    prompt = (
        "Create stable acceptance criteria for exactly this task. You are an independent read-only reviewer. "
        "You have not seen a design. Each criterion must quote an exact substring of the task. "
        "Keep this small (usually 4-8 criteria), checkable, and case-specific. Do not invent traditional "
        "topology, fabrication standards, dimensions, features, aesthetics, or physics requirements absent "
        "from the task. Retain all explicit numeric bounds and conditions. Appearance criteria cover only "
        "requested visible machine identity and detail. Function criteria cover the stated mechanical goal. "
        "These criteria will be frozen across revisions. Return the requested structured response.\n\nTASK:\n"+task)
    if construction_note and not initial_task_review:
        prompt = construction_note + prompt
    if initial_task_review:
        limits = ("active_budget_seconds", "simulation_timeout_seconds", "max_tasks_per_design", "max_parts")
        if not isinstance(tier_config, dict) or any(name not in tier_config for name in limits):
            raise ValueError("Initial task review requires the actual project budget limits")
        schema = copy.deepcopy(CRITERIA_SCHEMA)
        schema["properties"].update({
            "status":{"type":"string", "enum":["ready", "needs_revision", "unsupported"]},
            "task_text":STRING,
            "task_review":obj({"reasonable":{"type":"boolean"}, "reason":STRING}),
        })
        schema["required"].extend(["task_text", "task_review"])
        prompt = (
            "Review the initial task before any design exists. You are the Harness's independent "
            "read-only reviewer; the user explicitly delegates initial task reasonableness and "
            "selection of reasonable measurable initial acceptance indicators to you. Assess whether "
            "this task is a reasonable end-to-end mechanical generation task under the actual tier "
            "limits below. The budget is a resource constraint, not a prediction or guarantee of success. "
            "Choose initial acceptance indicators that the existing documented measurement interfaces "
            "and independent inspection of their recorded evidence can actually establish. Do not "
            "invent a requirement for a new bespoke measurement registry or assume unimplemented "
            "measurement capabilities. If the core function cannot be meaningfully verified with "
            "the available interfaces, report the specific gap rather than assuming future implementation. "
            "Read only this original task and the current generic Harness interface at "
            f"{CODE_ROOT/'simulation_ready_cad_workflow/validation_pipeline/slim_harness/README.md'}. "
            "Do not inspect existing designs or other project results.\n"
            "This task is explicitly adjustable. Preserve its machine identity, every core requested "
            "function (including both core functions when two are requested), and explicit prohibited "
            "means. You may select or revise initial measurable targets to make the task meaningful "
            "and testable within this tier. Do not remove a core function, waive verification, or relax "
            "the physical/geometry acceptance rules. The active-time budget, per-simulation timeout, "
            "maximum number of declared tests, and maximum number of actual physical parts are hard "
            "limits. Count independently made or purchased assembly parts, not rigid bodies or CAD "
            "solids. Do not require filler parts to satisfy a typical size range.\n"
            "Return task_text with the reviewed complete task, task_review.reasonable and its concrete "
            "reason, and stable criteria quoting exact substrings of that reviewed task_text. If the "
            "original is already appropriate, keep its text unchanged and still record your review. "
            "Return ready only when the reviewed task is reasonable and retains the core functions. "
            "If a meaningful in-scope task cannot yet be established, return needs_revision or "
            "unsupported, reasonable=false, and explain the unresolved scope or capability; do not "
            "pretend to approve it. Do not supply a mechanism architecture, topology, CAD recipe, "
            "implementation parameters, body/joint mapping, or implementation solution. Criteria "
            "describe observable outcomes, not a prescribed construction. This is task review only, "
            "not mechanical PASS or permission to bypass later independent review. The author still "
            "owns design and tests, and any later task tradeoff still needs its own independent review. "
            "Do not write files; the controller retains the original task and this review before "
            "applying approved text and criteria.\n"
            f"ACTUAL TIER CONFIGURATION:\n{json.dumps(tier_config,ensure_ascii=False)}\n"
            f"ORIGINAL TASK:\n{task}"
        )
    result = worker.run_turn(workspace,prompt,
        turn_dir,1200,readonly=True,fresh=True,response_schema=schema)
    if result["status"] != "ready": return result
    criteria_task = task
    if initial_task_review:
        review = result.get("task_review", {})
        criteria_task = result.get("task_text")
        if (review.get("reasonable") is not True or not isinstance(review.get("reason"), str)
                or not review["reason"].strip() or not isinstance(criteria_task, str) or not criteria_task.strip()):
            raise ValueError("Ready initial task review requires a reasonable task and a recorded reason")
    criteria = result["criteria"]
    if budget and sum(row.get('kind')=='function' for row in criteria)>budget.snapshot()['config']['max_tasks']:
        raise ValueError('Too many mechanical task groups; consolidate criteria without deleting core semantics')
    ids = [x["id"] for x in criteria]
    if not criteria or len(set(ids)) != len(ids) or any(not x["source_quote"] or x["source_quote"] not in criteria_task for x in criteria):
        raise ValueError("Independent criteria contain duplicate IDs or unsupported source quotes")
    if initial_task_review:
        return result
    frozen = {"schema_version":"slim_frozen_criteria_v1","task_file":str(task_path),
              "task_text":task,"criteria":criteria,"reviewer_summary":result["summary"]}
    if output.exists() and json.loads(output.read_text(encoding="utf-8")) != frozen:
        raise ValueError("Criteria already exist; a documented source correction is required")
    write_json_atomic(output,frozen)
    return result


def review_submission(worker, workspace, criteria, scene_path, evidence, turn_dir, *, physical=None, previous=None,
                      reports=(), static_evidence=(), static_errors=(), validation_scope=None, modeling_context=None,
                      program_evidence_paths=()):
    scene = json.loads(Path(scene_path).read_text(encoding="utf-8-sig"))
    test_ids = [test["id"] for test in scene["tests"]]
    response_schema = copy.deepcopy(REVIEW_SCHEMA)
    if program_evidence_paths:
        response_schema["properties"]["test_coverage"]["items"]["properties"]["program_governance_only"]={"type":"boolean"}
        response_schema["properties"]["test_coverage"]["items"]["required"].append("program_governance_only")
    from .review_phase import phase_schema, phase_prompt
    phase_schema(response_schema,criteria,final=physical is not None)
    if validation_scope is not None:
        validate_validation_scope(validation_scope)
        response_schema["properties"]["validation_scope_signature"] = {
            "type":"string", "enum":[validation_scope["signature"]],
            "description":"Exact program-supplied validation scope reviewed, including its explicit limitations.",
        }
        response_schema["required"].append("validation_scope_signature")
    response_schema["properties"]["test_coverage"]["items"]["properties"]["test_ids"] = {
        "type":"array", "items":{"type":"string", "enum":test_ids},
        "description":"Exact IDs from the current scene.tests only, never script paths or built-in check names.",
    }
    registered = {row.receipt_id: row for row in static_evidence if isinstance(row, VerifiedStaticEvidence)}
    static_properties = response_schema["properties"]["static_coverage"]["items"]["properties"]
    static_ids = [criterion["id"] for criterion in criteria["criteria"]
                  if is_static_delivery_criterion(criterion, static_evidence)
                  or has_static_evidence(criterion, static_evidence)]
    static_properties["criterion_id"] = {"type":"string", "enum":static_ids} if static_ids else STRING
    static_properties["receipt_ids"] = {"type":"array", "items":{"type":"string", "enum":list(registered)}} if registered else {"type":"array", "items":STRING, "maxItems":0}
    prompt = (
        "Independently review the delivered machine against the FROZEN criteria below. You are read-only. "
        "Review original-geometry images and actual files; appearance is not inferred from source code. "
        "Do not add requirements or demand customary topology if the delivered design meets the task. "
        "Batch concrete issues. For each relevant criterion report observed fact + evidence path and met, "
        "violated, or unobservable. Unclear views cause a specific evidence request, not a redesign demand. "
        "Before physics, observations cover appearance and every criterion with registered static evidence. "
        "Static receipts are program-owned completed executions bound to exact frozen criteria and current sources. "
        "Use static_coverage with only receipt IDs bound to that exact criterion, and cite a supplied receipt or "
        "inspection path in observed_fact/evidence_paths in BOTH phases. Copy receipt/inspection evidence_paths "
        "exactly as supplied, including the absolute path and its separators; a repository-relative equivalent "
        "does not satisfy the receipt binding. Independently inspect rebuilt/source/export correspondence, "
        "quantity-list consistency, actual original-geometry illustrations and documentation semantics. "
        "For legacy alarm F7 also check all four requested operating instructions. A receipt alone does not establish "
        "document semantics, actual insertability, material suitability or mechanical function. The supplied static_only "
        "mode can replace scene test coverage ONLY when the frozen criterion is exclusively static delivery; independently "
        "reject a mode that would exempt a mechanical claim. Omit static_only criteria entirely from test_coverage; "
        "even a row with empty test_ids is invalid. Use their static_coverage and observations instead. "
        "A supplemental receipt covers static obligations only and "
        "NEVER replaces nonempty appropriate scene test coverage for function criteria. Before physics its observation "
        "and verdict evaluate only the checked static portion: use met only when the independent static checks "
        "establish that portion, and explicitly state that future physical performance remains unverified. "
        "Do not mark that established static portion unobservable solely because physics has not run. "
        "If the static portion is itself unverified, report the concrete missing evidence. Final observations "
        "and verdicts must establish the whole criterion, including the actual physical results. "
        "With no applicable receipt, request concrete static evidence without rewriting criteria or inventing a motion test "
        "for export/document delivery. During preflight, absence of future physics results is EXPECTED and is never "
        "a reason to reject test coverage or request a model revision. Each test_coverage.test_ids entry "
        "must be an exact ID from AVAILABLE SCENE TEST IDS below, never a script path, measurement "
        "function, or built-in check name. Built-in original-shape collider, solver-step geometry, local "
        "connection, and duration checks are planned evidence associated with the relevant scene tests; "
        "describe their actual scope in reason. Supplied reports establish only the checks they record. "
        "A delivered script or source file is not proof that a check executed or passed; cite actual "
        "execution evidence in observations.evidence_paths before claiming completion. Do not assign an "
        "unrelated scene test just to fill a coverage row. If a frozen function criterion instead needs "
        "static delivery verification for which no registered static receipt is supplied, "
        "keep the criterion unchanged and return needs_evidence with a local_measurement request "
        "describing the missing Harness coverage interface. Do not invent a test ID, assume a script "
        "passed, or request a design revision unless concrete design evidence requires one. "
        "For a STEP-only source/export/inventory/document delivery gap, request action=local_measurement "
        "with exact target step_delivery:static_only for an exclusively static delivery criterion, or "
        "step_delivery:supplemental for a criterion mixing delivery and physical function. This invokes "
        "the controller's implemented verifier, not a request for printing artifacts. Its receipt "
        "establishes only executed rebuild/export/inventory checks and document existence; independently "
        "review assembly and document semantics, and retain motion-test coverage for mixed criteria. "
        "During preflight, ready means the declared scene tests and implemented backend can execute the accepted "
        "plan. During final review, ready requires actual physical evidence and satisfaction of every frozen criterion. "
        "Ordinary scene tests use the documented signal/assertion interface without a special measurement registration. "
        "A registered measurement is required only when relying on that specific installed measurement interface. "
        "If the author omitted or misused assertions expressible through supported signals and functions, return "
        "needs_revision with the concrete scene/test omission. Do not relabel author test coverage as a Harness gap. "
        "Conversely, when a required measurement cannot be expressed or executed by the supported interfaces, "
        "retain needs_evidence with the exact missing capability; an unexecuted author observer is not a substitute. "
        "Distinguish descriptive capabilities.json requests from installed executable interfaces. "
        "When PROGRAM MODELING CONTEXT below selects equivalent_elastic and confirms that its ABI is "
        "implemented, use that explicitly limited modeling scope. Do not demand a complete continuum "
        "finite-element model, deformed STEP spring audit or coil self-contact as a substitute goal. "
        "If the design still has only old capability requirements and omits the supported equivalent_elasticity "
        "declaration, material/geometry sources, assembly rest state, preload, passive coupling or working-range "
        "parameters, return needs_revision with concrete author-supplied omissions. Never invent those "
        "parameters. The existence of an interface does not prove a declaration is physically justified. "
        "If reasoning requires an API or measurement that the installed backend truly cannot execute, "
        "return needs_evidence and identify that actual Harness gap. Future physics outcomes are still "
        "expected to be absent at preflight. A conditional ready assuming future implementation is invalid. "
        "Reserve actual functional success/failure for the FINAL phase with physical summaries. "
        "Approve a proposed test only when duration, input, independent passive outputs, loads, "
        "and numeric assertions genuinely answer the task. Driving the output itself is not evidence of "
        "transmission. A spur_gears declaration uses finite equivalent tooth forces and sampled friction; "
        "it does not resolve individual tooth contact, backlash, impact, stress or deformation. Check that "
        "this representation and its declared parameter basis satisfy the frozen task; availability alone "
        "does not justify substituting it for required detailed contact. Original-shape motion audits remain "
        "required. Fixed bodies must correspond to justified external supports. Joint connection regions "
        "must stay local to real bearing/rail material, never span a disconnected machine as an invisible "
        "link. A source_support_checks radial_support_checked result verifies shaft/bore geometry only; "
        "independently check axial retention and the remaining constrained directions. An absent explicit "
        "support declaration is not an automatic failure: inspect the actual support using existing source evidence. "
        "This is physical behavior review, not matching joint or task templates. In the final phase "
        "also use the supplied physical results and solver-step original rigid-geometry audit within "
        "PROGRAM VALIDATION SCOPE. Original elastic CAD shown as static reference is not a deformation "
        "trace. Examine the named reference bodies and each pair-local flexible region; all other rigid "
        "features retain geometry/contact checks. The equivalent model does not establish spring winding "
        "clearance, local stress/life, or real clamps/slotted sleeves achieving its assumed preload. "
        "Actual stored energy, passive action/reaction, dissipation and registered contact measurements "
        "remain required. Only completed evidence within the declared scope establishes function. "
        "Echo its exact validation_scope_signature when supplied; never label it complete elastic "
        "source-geometry validation. Do not run simulation or modify anything. "
        "If previous observations are supplied, revisit only changed or unresolved evidence and carry "
        "unchanged supported observations forward; keep criteria stable.\n\n"
        f"SCENE: {scene_path}\nEVIDENCE: {json.dumps([str(x) for x in evidence])}\n"
        f"AVAILABLE SCENE TEST IDS: {json.dumps(test_ids,ensure_ascii=False)}\n"
        f"PROGRAM-VALIDATED STATIC RECEIPTS: {json.dumps([row.__dict__ for row in registered.values()],ensure_ascii=False)}\n"
        f"STATIC EVIDENCE GAPS: {json.dumps(list(static_errors),ensure_ascii=False)}\n"
        f"PROGRAM-OWNED GOVERNANCE EVIDENCE: {json.dumps([str(p) for p in program_evidence_paths])}\n"
        f"PROGRAM MODELING CONTEXT: {json.dumps(modeling_context,ensure_ascii=False)}\n"
        f"PROGRAM VALIDATION SCOPE: {json.dumps(validation_scope,ensure_ascii=False)}\n"
        f"AUTOMATIC GEOMETRY AND CONNECTION REPORTS: {json.dumps([str(x) for x in reports])}\n"
        f"CRITERIA:\n{json.dumps(criteria,ensure_ascii=False)}\n"
        f"PHYSICAL SUMMARY: {json.dumps(physical,ensure_ascii=False)}\n"
        f"PREVIOUS REVIEW: {json.dumps(previous,ensure_ascii=False)}"
    )
    from .construction import policy_prompt, construction_requested
    project=next((p for p in Path(scene_path).parents if (p/'construction_budget.json').is_file()),None)
    if project and construction_requested(project):
        plan_path=Path(scene_path).parent/'construction_plan.json'
        prompt+=policy_prompt(project)+'\nCONSTRUCTION PLAN CHANGES: '+(plan_path.read_text(encoding='utf-8') if plan_path.exists() else 'none')
        prompt+='\nJudge core functional meaning, not maximum possible fidelity. Approve explicitly scoped flexible/abstract joints with static original-geometry evidence when the required function is actually tested; never claim detailed contact, strength or deformation that was not solved. Missing optional enhancements should not block a core-complete construction candidate. Administrative requirements (time accounting, completion summary, source delivery lists) are generated/checked by the Harness separately; do not require a simulation assertion to prove them or require a completion summary before final mechanical review. Retain every mechanical core obligation.\n'
    if program_evidence_paths:
        prompt += (
            "\nUse the supplied current governance snapshot for active/wall accounting, verified user budget "
            "grants, test/physical-part inventory and task-adjustment history. A separately authorized recovery "
            "grant changes the continuation's effective total budget (base plus verified grants); it does not "
            "rewrite the original one-hour result, reset used active time, excuse historical failures, or change "
            "the per-simulation timeout, test count or physical-part caps. State original-budget outcomes and "
            "authorized continuation separately. Do not reject an authorized continuation solely because total "
            "active exceeds the original base while remaining within its verified effective limit. "
            "In remote-job envelopes, effective_native_limit_seconds is the granted execution allowance; "
            "native_wall_seconds is measured after process and pipe cleanup and is not the kill timestamp. "
            "Do not infer native deadline failure or exact compliance solely from that total; use the actual "
            "watchdog/termination evidence and state when the precise termination time was not recorded. "
            "For a criterion that is EXCLUSIVELY program governance (budget/resource limits, inventory counts, "
            "or unchanged-task/adjustment history), you may set program_governance_only=true in its "
            "test_coverage row, appropriate=true and test_ids=[]. Its observation must be met with concrete "
            "facts citing the supplied governance snapshot. Do not invent a motion test for bookkeeping. "
            "Never use this flag for a mechanical function or a mixed physical/governance criterion: those "
            "must set program_governance_only=false and still require appropriate real scene-test coverage "
            "and final physical evidence. Set false for every other non-governance coverage row. The snapshot "
            "does not establish source rebuild, mechanical function, motion legality or assembly correctness.\n"
        )
    prompt=phase_prompt(criteria,final=physical is not None)+prompt
    return worker.run_turn(workspace,prompt,turn_dir,1200,images=evidence,readonly=True,fresh=True,response_schema=response_schema)


def _static_contract_errors(result, criteria, static_evidence=()):
    rows = result.get("static_coverage", [])
    if len({row["criterion_id"] for row in rows}) != len(rows):
        return ["static_coverage: duplicate criterion rows"]
    allowed = {criterion["id"] for criterion in criteria["criteria"]
               if is_static_delivery_criterion(criterion, static_evidence)
               or has_static_evidence(criterion, static_evidence)}
    if any(row["criterion_id"] not in allowed for row in rows):
        return ["static_coverage: criterion has no applicable registered static execution evidence"]
    return []


def _criterion_static_errors(result, criterion, static_evidence):
    key = criterion["id"]
    registered = {row.receipt_id:row for row in static_evidence if applies(row, criterion)}
    row = next((row for row in result.get("static_coverage", []) if row["criterion_id"] == key), {})
    identifiers = row.get("receipt_ids", [])
    errors = []
    if not identifiers or len(set(identifiers)) != len(identifiers) or not set(identifiers) <= set(registered):
        errors.append(f"static_coverage[{key}]: requires nonempty, unique, program-validated execution receipt IDs for this criterion")
    if is_static_delivery_criterion(criterion, static_evidence) and any(row["criterion_id"] == key for row in result.get("test_coverage", [])):
        errors.append(f"test_coverage[{key}]: a scene test cannot substitute for static delivery execution")
    observation = next((row for row in result.get("observations", []) if row["criterion_id"] == key), {})
    executed_paths = {path for identifier in identifiers if identifier in registered for path in registered[identifier].evidence_paths[:2]}
    if not executed_paths.intersection(observation.get("evidence_paths", [])):
        errors.append(f"observations[{key}]: cite an exact registered receipt or inspection path, plus independent semantic facts")
    return errors


def _scope_contract_errors(result, scene, validation_scope):
    if validation_scope is None:
        return ["validation_scope: equivalent scene requires a compiled scope"] if scene.get("equivalent_elasticity") else []
    try:
        validate_validation_scope(validation_scope)
    except ValidationScopeError as exc:
        return ["validation_scope: " + str(exc)]
    if result.get("validation_scope_signature") != validation_scope["signature"]:
        return ["validation_scope_signature: ready requires the exact supplied modeling scope"]
    return []


def review_failure_reasons(result, criteria, scene, *, final=False, static_evidence=(), validation_scope=None,
                           program_evidence_paths=()):
    """Explain the existing review gate without replacing or relaxing it."""
    reasons = _static_contract_errors(result, criteria, static_evidence)+_scope_contract_errors(result,scene,validation_scope)
    observations = {row["criterion_id"]:row for row in result.get("observations",[])}
    coverage = {row["criterion_id"]:row for row in result.get("test_coverage",[])}
    for field, rows in (("observations",observations),("test_coverage",coverage)):
        if len(rows) != len(result.get(field,[])):
            reasons.append(f"{field}: duplicate criterion rows")
    tests = {test["id"] for test in scene["tests"]}
    program_paths={str(path) for path in program_evidence_paths if Path(path).is_file()}
    for criterion in criteria["criteria"]:
        key = criterion["id"]
        static_only = is_static_delivery_criterion(criterion, static_evidence)
        registered_static = has_static_evidence(criterion, static_evidence)
        program_only=coverage.get(key,{}).get("program_governance_only") is True
        if final or criterion["kind"] == "appearance" or static_only or registered_static or program_only:
            row = observations.get(key,{})
            missing = [field for field in ("observed_fact","evidence_paths") if not row.get(field)]
            if row.get("verdict") != "met" or missing:
                reasons.append(f"observations[{key}]: ready requires met with observed_fact and evidence_paths")
        if static_only or registered_static:
            reasons.extend(_criterion_static_errors(result, criterion, static_evidence))
        if static_only:
            continue
        if criterion["kind"] == "function":
            row = coverage.get(key,{})
            if program_only:
                observation=observations.get(key,{})
                if (not program_paths or row.get("test_ids") or row.get("appropriate") is not True
                        or not row.get("reason") or not program_paths.intersection(observation.get("evidence_paths",[]))):
                    reasons.append(f"test_coverage[{key}]: program-only coverage requires an explicit independent reason, no motion test, and observation citing current controller governance evidence")
                continue
            if not row.get("appropriate") or not row.get("test_ids"):
                reasons.append(f"test_coverage[{key}]: ready requires appropriate coverage and nonempty scene test IDs")
            unknown = sorted(set(row.get("test_ids",[]))-tests)
            if unknown:
                reasons.append(f"test_coverage[{key}]: unknown scene test IDs {unknown}; available IDs {sorted(tests)}")
    return reasons


def review_passes(result, criteria, scene, *, final=False, static_evidence=(), validation_scope=None, program_evidence_paths=()):
    return result.get("status") == "ready" and not review_failure_reasons(
        result, criteria, scene, final=final, static_evidence=static_evidence, validation_scope=validation_scope,
        program_evidence_paths=program_evidence_paths)
