"""Frozen requirement obligations and exact task-plan coverage checks.

The Harness never infers mechanical meaning from prompt prose here.  Numeric
acceptance rows are already operator-authored structured data.  Non-numeric
and visual obligations use an explicit object form with stable identifiers,
named checks, evidence roles, and (when needed) implementation capabilities.
Legacy strings can still be inventoried, but a new plan cannot certify them
until an operator replaces them with the explicit form.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import re
from typing import Any

from ..mechanism.io import canonical_sha256
from .errors import ReactLoopError
from .payloads import strict_json_copy


OBLIGATION_INVENTORY_SCHEMA = "simulation_ready_cad_requirement_obligations_v1"
OBLIGATION_COVERAGE_SCHEMA = "simulation_ready_cad_requirement_coverage_v1"

_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_NAMESPACED_ID = re.compile(
    r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$"
)
_KINDS = frozenset({"acceptance", "engineering", "visual"})
_COVERAGE_MODES = frozenset(
    {"validator", "saved_state_audit", "reviewer_visual", "policy"}
)
OBLIGATION_CHECK_REGISTRY: dict[str, str] = {
    "validator.actuation.indirect_only": "validator",
    "validator.mechanical.fixed_chassis_and_moving_groups": "validator",
    "validator.mechanical.unique_input_and_output": "validator",
    "validator.mechanical.finite_support_and_transmission": "validator",
    "validator.mechanical.audit_geometry_exposed": "validator",
    "validator.mechanical.single_source_cad": "validator",
    "validator.mechanical.rigid_group_ownership": "validator",
    "validator.mechanical.saved_pose_legality": "saved_state_audit",
    "validator.clock.independent_alarm_power_paths": "validator",
    "validator.chain.moving_path_state": "saved_state_audit",
    "validator.chain.saved_state_clearance": "saved_state_audit",
    "validator.bicycle.native_ground_contact": "validator",
    "validator.bicycle.front_wheel_passive": "validator",
    "validator.bicycle.ground_roll_step_geometry": "validator",
    "validator.bicycle.ground_roll_behavior": "validator",
    "dashboard.chain.phase_animation": "saved_state_audit",
    "dashboard.pose.index_change": "saved_state_audit",
    "reviewer.visual.complete_bicycle": "reviewer_visual",
    "reviewer.visual.drivetrain_visible": "reviewer_visual",
    "reviewer.visual.complete_detailed_bicycle": "reviewer_visual",
    "reviewer.visual.steering_motion": "reviewer_visual",
    "reviewer.visual.chain_drivetrain_motion": "reviewer_visual",
    "reviewer.visual.full_machine_ground_roll": "reviewer_visual",
    "reviewer.visual.replay_consistency": "reviewer_visual",
    "reviewer.visual.complete_twin_bell_clock": "reviewer_visual",
    "reviewer.visual.clock_three_hand_motion": "reviewer_visual",
    "reviewer.visual.clock_manual_setting": "reviewer_visual",
    "reviewer.visual.clock_independent_alarm": "reviewer_visual",
    "reviewer.visual.clock_alarm_target_setting": "reviewer_visual",
    "reviewer.visual.clock_selected_position_alarm": "reviewer_visual",
    "policy.bicycle.guided_test_boundaries": "policy",
    "policy.clock.validation_boundaries": "policy",
    "policy.visual.informative_only": "policy",
}
OBLIGATION_CHECK_DETAILS: dict[str, tuple[str, str, str]] = {
    "validator.actuation.indirect_only": (
        "The saved actuation provenance and wrench ledger for the task target.",
        "No external actuator is attached to the output; its motion is caused only by the declared input and transmission reactions.",
        "Typed task metric evidence and the saved wrench audit.",
    ),
    "validator.mechanical.fixed_chassis_and_moving_groups": (
        "The STEP-bound body-role assignment and every saved body pose.",
        "The declared chassis remains fixed while the declared input and output groups remain the moving bodies used by the task.",
        "Mechanism role binding, collision-body manifest, and physical body-pose trace.",
    ),
    "validator.mechanical.unique_input_and_output": (
        "Typed input-channel, task-observable, and external-wrench bindings.",
        "Exactly the declared crank is powered and exactly the declared rear-wheel observable is evaluated as the task target.",
        "Semantic binding, acceptance contract, and saved wrench audit.",
    ),
    "validator.mechanical.finite_support_and_transmission": (
        "Saved support/transmission wrenches and conditional physical-interface evidence.",
        "All required physical reactions are present and finite, and no ideal joint supplies authority to the physical branch.",
        "Saved wrench audit, interface declarations, geometry gates, and conditional-interface runtime audit.",
    ),
    "validator.mechanical.audit_geometry_exposed": (
        "STEP-to-body/role bindings and the original per-solid collision inventory.",
        "All governed input, output, support, sprocket, and transmission geometry is hash-bound and available to the independent audits.",
        "Intake bindings, functional-geometry evidence, and original STEP collision components.",
    ),
    "validator.mechanical.single_source_cad": (
        "The authoritative STEP binding, functional-geometry report, and per-solid collision inventory.",
        "Visible and physical geometry resolves to the same package-bound STEP assembly rather than a dashboard-only substitute.",
        "Certificate-bound comparison, functional-geometry evidence, and original STEP collision components.",
    ),
    "validator.mechanical.rigid_group_ownership": (
        "The five declared rigid bodies, their unique group membership, collision components, and saved poses.",
        "Every declared group belongs to exactly one rigid body and every saved physical sample carries the complete body set.",
        "Certificate-bound comparison, collision-body inventory, and physical body-pose trace.",
    ),
    "validator.mechanical.saved_pose_legality": (
        "Every saved physical clock pose against the authoritative STEP component geometry.",
        "All saved poses pass the bounded geometry legality audit without a forbidden collision, separation, or unsupported interface state.",
        "Saved-pose geometry audit and original STEP collision components.",
    ),
    "validator.clock.independent_alarm_power_paths": (
        "The clock family topology, input provenance, and saved timekeeping and alarm traces.",
        "The time barrel drives all three hands, the setting input acts through its clutch, and a distinct alarm barrel drives the cam and hammer without directly actuating a hand.",
        "Certificate-bound clock comparison, branch semantics, and saved wrench audit.",
    ),
    "validator.chain.moving_path_state": (
        "The validator-owned fixed chain path and every saved material-phase state.",
        "The contract re-derives exactly, every sample is hash-bound to it, and phase/force coordinates agree with the two sprocket angles.",
        "Moving-chain path contract and per-sample moving-chain state audit.",
    ),
    "validator.chain.saved_state_clearance": (
        "The moving chain tube against original STEP component meshes at every saved physical state.",
        "No chain region intersects any component. The matching sprocket is permitted only inside its own fixed authored-world flex zone, and its measured penetration may not exceed the public chain-and-sprocket-geometry-derived cap.",
        "Saved-state moving-chain clearance audit and original STEP component hashes.",
    ),
    "validator.bicycle.native_ground_contact": (
        "The physical engine/link declaration, contact model, and paired wheel-ground callbacks.",
        "Both wheels use native finite Coulomb contact with no hidden rolling constraint, pose setter, or contact-bypass ideal link.",
        "Certificate-bound comparison and saved physical contact trace.",
    ),
    "validator.bicycle.front_wheel_passive": (
        "The branch semantics and saved external-wrench audit for the front wheel.",
        "Only the rear wheel receives task torque and the front wheel remains passive while the chassis is free to translate.",
        "Certificate-bound comparison, body-pose trace, and wrench audit.",
    ),
    "validator.bicycle.ground_roll_step_geometry": (
        "The STEP-bound wheel radii, axle contact heights, finite ground plane, and visible upright guide.",
        "Both wheel contacts and the guide resolve to the submitted package-bound STEP solids before physics.",
        "Certificate-bound comparison, initial functional geometry, and original STEP collision components.",
    ),
    "validator.bicycle.ground_roll_behavior": (
        "The full-cadence physical samples over settle, powered travel, and passive observation.",
        "Travel, passive front rotation, slip, paired contact coverage, separation, drift, off-axis rotation, and penetration all satisfy the public thresholds.",
        "Certificate-bound comparison, contact trace, and saved-pose geometry audit.",
    ),
    "dashboard.chain.phase_animation": (
        "The dashboard replay's chain markers and their saved phase binding.",
        "The replay uses only a valid hash-bound path/state and visibly advances material phase along the fixed spatial path.",
        "Dashboard physics replay generated from saved physical trace frames.",
    ),
    "dashboard.pose.index_change": (
        "Saved crank and rear-wheel orientations used by the replay.",
        "Powered and returned frames contain the required non-zero index-mark pose changes and are not fabricated by the renderer.",
        "Saved body-pose trace and dashboard physics replay.",
    ),
    "reviewer.visual.complete_bicycle": (
        "The rendered STEP assembly as a whole.",
        "An independent reviewer can identify two aligned wheels, diamond frame, fork, handlebar, saddle, and a complete conventional bicycle silhouette.",
        "Non-certifying design 3D review evidence.",
    ),
    "reviewer.visual.drivetrain_visible": (
        "The rendered crank, pedals, chainring, chain path, and rear sprocket.",
        "An independent reviewer can distinguish every listed drivetrain element and see the chain path engage both sprockets.",
        "Non-certifying design 3D and physics-replay evidence.",
    ),
    "reviewer.visual.complete_detailed_bicycle": (
        "Opposed isometric, orthographic, and focused interface views of the authoritative STEP assembly.",
        "An independent reviewer sees complete, plausible construction rather than a coarse bicycle silhouette or drivetrain rig.",
        "Non-certifying design 3D review evidence bound to the current STEP.",
    ),
    "reviewer.visual.steering_motion": (
        "The steering-task replay and its handlebar, fork, and front-wheel poses.",
        "The complete steering assembly visibly turns through the required sweep without detachment or impossible intersection.",
        "Non-certifying physics replay generated from the steering trace.",
    ),
    "reviewer.visual.chain_drivetrain_motion": (
        "The crank-drive replay including pedals, crank, chain material phase, sprockets, and rear wheel.",
        "All drivetrain elements visibly advance together and their index orientations change consistently with the saved trace.",
        "Non-certifying physics replay generated from the chain trace.",
    ),
    "reviewer.visual.full_machine_ground_roll": (
        "The ground-roll replay including ground, guide, chassis, steering, rear wheel, and front wheel.",
        "The non-ground bicycle translates as one machine while both wheels rotate and the unactuated front wheel rolls passively.",
        "Non-certifying physics replay generated from the ground-roll trace.",
    ),
    "reviewer.visual.replay_consistency": (
        "The interactive real-physics replays and paired physical-versus-ideal certification views for every public task.",
        "Both dashboard sections expose the same task set and the same moving bodies from their canonical traces.",
        "Non-certifying dashboard replay and certificate-video evidence.",
    ),
    "reviewer.visual.complete_twin_bell_clock": (
        "Opposed isometric, front, side, rear, and opened or section views of the authoritative STEP clock.",
        "An independent reviewer can identify a complete conventional twin-bell alarm clock, including its housing, dial, three hands, bells, hammer, rear controls, and plausible internal construction.",
        "Non-certifying design 3D review evidence bound to the current STEP.",
    ),
    "reviewer.visual.clock_three_hand_motion": (
        "The timekeeping replay and its hour, minute, and second hand poses.",
        "All three hands visibly move together in the declared display direction through the authored clock train.",
        "Non-certifying physics replay generated from the timekeeping trace.",
    ),
    "reviewer.visual.clock_manual_setting": (
        "The manual-setting replay, including the rear setting control, finite clutch, and all three hands.",
        "The rear control visibly sets the display through the authored clutch and gear train without direct hand animation.",
        "Non-certifying physics replay generated from the manual-setting trace.",
    ),
    "reviewer.visual.clock_independent_alarm": (
        "The isolated alarm replay, including the alarm barrel, cam, follower, hammer, bells, and all three hands.",
        "The hammer visibly rings between the bells while the three-hand display remains effectively stationary.",
        "Non-certifying physics replay generated from the isolated-alarm trace.",
    ),
    "reviewer.visual.clock_alarm_target_setting": (
        "The alarm-setting replay, including the hour hand, its finite clutch, and the dedicated retained alarm-target cam.",
        "Turning the hour hand visibly moves and latches the separate alarm target without direct target-cam animation.",
        "Non-certifying physics replay generated from the hour-hand alarm-setting trace.",
    ),
    "reviewer.visual.clock_selected_position_alarm": (
        "The selected-position alarm replay from target setting through quiet advance, ringing, and continued timekeeping.",
        "The alarm remains quiet before the hour hand reaches the retained target, then rings while the display continues advancing.",
        "Non-certifying physics replay generated from the selected-position alarm trace.",
    ),
    "policy.bicycle.guided_test_boundaries": (
        "The ground-roll comparison's strict nonclaims and declared visible-guide mode.",
        "The result claims only guided upright straight rolling and does not claim balance, tire deformation or wear, strength, fatigue, or steering dynamics.",
        "Certificate-bound ground-roll comparison and terminal verdict.",
    ),
    "policy.clock.validation_boundaries": (
        "The certified clock comparisons and their strict nonclaims.",
        "The result is limited to the frozen guided clock functions, ratios, alarm isolation, and one-indicated-minute sequence and does not claim real-time accuracy, endurance, wear, manufacturability, or consumer safety.",
        "Certificate-bound clock comparisons and terminal verdict.",
    ),
    "policy.visual.informative_only": (
        "The final verdict authority and dashboard evidence metadata.",
        "No visual render or reviewer impression is used as mechanical or physical PASS authority.",
        "Terminal validation result and dashboard non-authority declarations.",
    ),
}

if set(OBLIGATION_CHECK_DETAILS) != set(OBLIGATION_CHECK_REGISTRY):
    raise RuntimeError("requirement check registry and documentation differ")


class RequirementObligationError(ReactLoopError):
    def __init__(self, code: str, message: str, *, json_path: str = "$") -> None:
        self.json_path = json_path
        super().__init__(code, message)


def _fail(code: str, message: str, *, json_path: str = "$") -> None:
    raise RequirementObligationError(code, message, json_path=json_path)


def _text(value: Any, *, label: str, limit: int = 8_192) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        _fail(
            "REQUIREMENT_OBLIGATION_INVALID",
            f"{label} must be a non-empty bounded string",
            json_path=label,
        )
    return value.strip()


def _identifier(value: Any, *, label: str) -> str:
    result = _text(value, label=label, limit=128)
    if _IDENTIFIER.fullmatch(result) is None:
        _fail(
            "REQUIREMENT_OBLIGATION_INVALID",
            f"{label} must be a stable identifier",
            json_path=label,
        )
    return result


def _namespaced_list(
    value: Any,
    *,
    label: str,
    allow_empty: bool = True,
) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) > 128
        or (not allow_empty and not value)
    ):
        _fail(
            "REQUIREMENT_OBLIGATION_INVALID",
            f"{label} must be a bounded list",
            json_path=label,
        )
    output: list[str] = []
    for index, item in enumerate(value):
        text = _text(item, label=f"{label}[{index}]", limit=256)
        if _NAMESPACED_ID.fullmatch(text) is None:
            _fail(
                "REQUIREMENT_OBLIGATION_INVALID",
                f"{label}[{index}] must be a namespaced lowercase identifier",
                json_path=f"{label}[{index}]",
            )
        output.append(text)
    if len(set(output)) != len(output):
        _fail(
            "REQUIREMENT_OBLIGATION_INVALID",
            f"{label} contains duplicate identifiers",
            json_path=label,
        )
    return output


def _identifier_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > 128:
        _fail(
            "REQUIREMENT_OBLIGATION_INVALID",
            f"{label} must be a non-empty bounded list",
            json_path=label,
        )
    output = [
        _identifier(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    ]
    if len(set(output)) != len(output):
        _fail(
            "REQUIREMENT_OBLIGATION_INVALID",
            f"{label} contains duplicate identifiers",
            json_path=label,
        )
    return output


def _registered_check_mode(check_id: str, *, label: str) -> str:
    prefix = "validator.metric."
    if check_id.startswith(prefix):
        metric_id = check_id[len(prefix) :]
        from ..intake.acceptance_contract import METRIC_REGISTRY
        from ..intake.clock_task_suite import CLOCK_TASK_SUITE_EVALUATOR_IDS
        from ..intake.egg_beater_task_suite import (
            EGG_BEATER_TASK_SUITE_EVALUATOR_IDS,
        )
        from ..intake.rack_pinion_gate_task_suite import (
            RACK_PINION_GATE_TASK_SUITE_EVALUATOR_IDS,
        )
        from ..intake.tongs_task_suite import TONGS_TASK_SUITE_EVALUATOR_IDS
        from ..intake.utility_knife_task_suite import (
            UTILITY_KNIFE_TASK_SUITE_EVALUATOR_IDS,
        )

        if (
            metric_id in METRIC_REGISTRY
            or metric_id in CLOCK_TASK_SUITE_EVALUATOR_IDS
            or metric_id in EGG_BEATER_TASK_SUITE_EVALUATOR_IDS
            or metric_id in UTILITY_KNIFE_TASK_SUITE_EVALUATOR_IDS
            or metric_id in TONGS_TASK_SUITE_EVALUATOR_IDS
            or metric_id in RACK_PINION_GATE_TASK_SUITE_EVALUATOR_IDS
        ):
            return "validator"
    mode = OBLIGATION_CHECK_REGISTRY.get(check_id)
    if mode is None:
        _fail(
            "REQUIREMENT_OBLIGATION_CHECK_UNKNOWN",
            f"{label} references unregistered Harness check {check_id!r}",
            json_path=label,
        )
    return mode


def _legacy_obligation_id(kind: str, index: int, statement: str) -> str:
    digest = hashlib.sha256(statement.encode("utf-8")).hexdigest()[:12]
    return f"legacy.{kind}.{index + 1:03d}.{digest}"


def _explicit_obligation(
    value: Any,
    *,
    kind: str,
    path: str,
    index: int,
) -> dict[str, Any]:
    if isinstance(value, str):
        statement = _text(value, label=path)
        core = {
            "obligation_id": _legacy_obligation_id(kind, index, statement),
            "kind": kind,
            "source_path": path,
            "statement": statement,
            "classification_status": "LEGACY_UNCLASSIFIED",
            "coverage_mode": None,
            "required_check_ids": [],
            "required_capability_ids": [],
            "required_evidence_roles": [],
            "acceptance_binding": None,
        }
        return {**core, "source_canonical_sha256": canonical_sha256(value)}
    required = {
        "obligation_id",
        "statement",
        "coverage_mode",
        "required_check_ids",
        "required_capability_ids",
        "required_evidence_roles",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        _fail(
            "REQUIREMENT_OBLIGATION_INVALID",
            f"{path} must be a legacy string or an explicit closed obligation object",
            json_path=path,
        )
    obligation_id = _identifier(
        value.get("obligation_id"), label=f"{path}.obligation_id"
    )
    mode = _text(value.get("coverage_mode"), label=f"{path}.coverage_mode", limit=32)
    if mode not in _COVERAGE_MODES:
        _fail(
            "REQUIREMENT_OBLIGATION_INVALID",
            f"{path}.coverage_mode is unsupported",
            json_path=f"{path}.coverage_mode",
        )
    checks = _namespaced_list(
        value.get("required_check_ids"),
        label=f"{path}.required_check_ids",
        allow_empty=False,
    )
    check_modes = {
        _registered_check_mode(check_id, label=f"{path}.required_check_ids")
        for check_id in checks
    }
    if mode not in check_modes:
        _fail(
            "REQUIREMENT_OBLIGATION_CHECK_MODE_MISMATCH",
            f"{path} has no registered {mode!r} check",
            json_path=f"{path}.required_check_ids",
        )
    evidence_roles = _namespaced_list(
        value.get("required_evidence_roles"),
        label=f"{path}.required_evidence_roles",
        allow_empty=False,
    )
    capabilities = _namespaced_list(
        value.get("required_capability_ids"),
        label=f"{path}.required_capability_ids",
    )
    core = {
        "obligation_id": obligation_id,
        "kind": kind,
        "source_path": path,
        "statement": _text(value.get("statement"), label=f"{path}.statement"),
        "classification_status": "EXPLICIT",
        "coverage_mode": mode,
        "required_check_ids": checks,
        "required_capability_ids": capabilities,
        "required_evidence_roles": evidence_roles,
        "acceptance_binding": None,
    }
    return {**core, "source_canonical_sha256": canonical_sha256(value)}


def build_requirement_obligation_inventory(
    requirements: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a detached, hash-bound inventory without interpreting prose."""

    if not isinstance(requirements, Mapping):
        _fail("REQUIREMENT_OBLIGATION_INVALID", "requirements must be an object")
    entries: list[dict[str, Any]] = []
    tasks = requirements.get("tasks")
    if not isinstance(tasks, list):
        _fail("REQUIREMENT_OBLIGATION_INVALID", "requirements.tasks must be a list")
    for task_index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            continue
        acceptance = task.get("acceptance_requirements")
        if not isinstance(acceptance, list):
            continue
        for item_index, item in enumerate(acceptance):
            path = f"$.tasks[{task_index}].acceptance_requirements[{item_index}]"
            if not isinstance(item, Mapping):
                _fail(
                    "REQUIREMENT_OBLIGATION_INVALID",
                    f"{path} must be an object",
                    json_path=path,
                )
            obligation_id = _identifier(
                item.get("requirement_id"),
                label=f"{path}.requirement_id",
            )
            metric_id = _text(item.get("metric_id"), label=f"{path}.metric_id", limit=256)
            load_case_id = _text(
                item.get("test_load_case_id"),
                label=f"{path}.test_load_case_id",
                limit=256,
            )
            branches = item.get("required_branches")
            if not isinstance(branches, list) or not branches:
                _fail(
                    "REQUIREMENT_OBLIGATION_INVALID",
                    f"{path}.required_branches must be non-empty",
                    json_path=f"{path}.required_branches",
                )
            normalized_branches = [
                _text(branch, label=f"{path}.required_branches[{index}]", limit=64)
                for index, branch in enumerate(branches)
            ]
            required_checks = [f"validator.metric.{metric_id}"]
            if item.get("allow_direct_output_actuation") is False:
                required_checks.append("validator.actuation.indirect_only")
            core = {
                "obligation_id": obligation_id,
                "kind": "acceptance",
                "source_path": path,
                "statement": _text(
                    item.get("prompt_clause"), label=f"{path}.prompt_clause"
                ),
                "classification_status": "EXPLICIT",
                "coverage_mode": "validator",
                "required_check_ids": required_checks,
                "required_capability_ids": [],
                "required_evidence_roles": ["validation.task_metric_evidence"],
                "acceptance_binding": {
                    "metric_id": metric_id,
                    "test_load_case_id": load_case_id,
                    "required_branches": normalized_branches,
                    "allow_direct_output_actuation": item.get(
                        "allow_direct_output_actuation"
                    ),
                },
            }
            entries.append(
                {**core, "source_canonical_sha256": canonical_sha256(item)}
            )
    for field, kind in (
        ("non_numeric_engineering_constraints", "engineering"),
        ("visual_evidence_checklist", "visual"),
    ):
        values = requirements.get(field, [])
        if not isinstance(values, list):
            _fail(
                "REQUIREMENT_OBLIGATION_INVALID",
                f"$.{field} must be a list when present",
                json_path=f"$.{field}",
            )
        for index, value in enumerate(values):
            entries.append(
                _explicit_obligation(
                    value,
                    kind=kind,
                    path=f"$.{field}[{index}]",
                    index=index,
                )
            )
    seen: set[str] = set()
    for entry in entries:
        obligation_id = entry["obligation_id"]
        if obligation_id in seen:
            _fail(
                "REQUIREMENT_OBLIGATION_DUPLICATE",
                f"obligation_id {obligation_id!r} is duplicated",
                json_path=entry["source_path"],
            )
        seen.add(obligation_id)
    core = {
        "schema_version": OBLIGATION_INVENTORY_SCHEMA,
        "requirements_id": _text(
            requirements.get("requirements_id"), label="$.requirements_id", limit=256
        ),
        "requirements_canonical_sha256": canonical_sha256(requirements),
        "entries": entries,
    }
    return strict_json_copy(
        {**core, "canonical_payload_sha256": canonical_sha256(core)},
        label="requirement obligation inventory",
    )


def _selected_joint_capabilities(
    *,
    plan_item_ids: set[str],
    template_selections: Sequence[Mapping[str, Any]],
    catalog_snapshot: Mapping[str, Any],
) -> set[str]:
    entries = {
        (
            str(item.get("template_id")),
            str(item.get("version")),
            str(item.get("sha256")),
        ): item
        for item in catalog_snapshot.get("entries", [])
        if isinstance(item, Mapping) and item.get("kind") == "joint"
    }
    output: set[str] = set()
    for selection in template_selections:
        if (
            selection.get("kind") != "joint"
            or str(selection.get("plan_item_id")) not in plan_item_ids
        ):
            continue
        record = entries.get(
            (
                str(selection.get("template_id")),
                str(selection.get("version")),
                str(selection.get("sha256")),
            )
        )
        if not isinstance(record, Mapping):
            continue
        variant = next(
            (
                item
                for item in record.get("capability_variants", [])
                if isinstance(item, Mapping)
                and item.get("variant_id") == selection.get("selection_id")
            ),
            None,
        )
        if isinstance(variant, Mapping):
            output.update(
                str(value) for value in variant.get("capability_ids", [])
            )
    return output


def validate_obligation_coverage(
    *,
    inventory: Mapping[str, Any],
    coverage: Any,
    joint_plan: Sequence[Mapping[str, Any]],
    test_plan: Sequence[Mapping[str, Any]],
    template_selections: Sequence[Mapping[str, Any]],
    catalog_snapshot: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return canonical coverage or reject any gap, unknown ID, or mismatch."""

    entries = inventory.get("entries") if isinstance(inventory, Mapping) else None
    if not isinstance(entries, list) or inventory.get("schema_version") != OBLIGATION_INVENTORY_SCHEMA:
        raise ReactLoopError(
            "REACT_REQUIREMENT_OBLIGATIONS_INVALID",
            "the frozen requirement obligation inventory is invalid",
        )
    if not isinstance(coverage, list) or len(coverage) != len(entries):
        raise ReactLoopError(
            "REACT_REQUIREMENT_COVERAGE_INCOMPLETE",
            "obligation_coverage must contain exactly one row for every frozen obligation",
        )
    inventory_by_id = {
        str(item["obligation_id"]): item
        for item in entries
        if isinstance(item, Mapping)
    }
    joint_ids = {str(item.get("joint_id")) for item in joint_plan}
    test_by_id = {str(item.get("test_id")): item for item in test_plan}
    plan_ids = joint_ids | set(test_by_id)
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    required_fields = {
        "obligation_id",
        "plan_item_ids",
        "model_element_ids",
        "check_ids",
        "evidence_roles",
        "failure_diagnostic",
    }
    for index, raw in enumerate(coverage):
        label = f"obligation_coverage[{index}]"
        if not isinstance(raw, Mapping) or set(raw) != required_fields:
            raise ReactLoopError(
                "REACT_REQUIREMENT_COVERAGE_INVALID",
                f"{label} must contain exactly {sorted(required_fields)}",
            )
        obligation_id = _identifier(raw.get("obligation_id"), label=f"{label}.obligation_id")
        if obligation_id in seen or obligation_id not in inventory_by_id:
            raise ReactLoopError(
                "REACT_REQUIREMENT_COVERAGE_UNKNOWN",
                f"{label} repeats or references unknown obligation {obligation_id!r}",
            )
        seen.add(obligation_id)
        plan_items = _identifier_list(
            raw.get("plan_item_ids"), label=f"{label}.plan_item_ids"
        )
        if any(item not in plan_ids for item in plan_items):
            raise ReactLoopError(
                "REACT_REQUIREMENT_COVERAGE_PLAN_ITEM_UNKNOWN",
                f"{label} references a plan item that does not exist",
            )
        model_elements = _namespaced_list(
            raw.get("model_element_ids"),
            label=f"{label}.model_element_ids",
            allow_empty=False,
        )
        check_ids = _namespaced_list(
            raw.get("check_ids"), label=f"{label}.check_ids", allow_empty=False
        )
        for check_id in check_ids:
            _registered_check_mode(check_id, label=f"{label}.check_ids")
        evidence_roles = _namespaced_list(
            raw.get("evidence_roles"),
            label=f"{label}.evidence_roles",
            allow_empty=False,
        )
        obligation = inventory_by_id[obligation_id]
        missing_checks = set(obligation["required_check_ids"]) - set(check_ids)
        missing_evidence = set(obligation["required_evidence_roles"]) - set(evidence_roles)
        capabilities = _selected_joint_capabilities(
            plan_item_ids=set(plan_items),
            template_selections=template_selections,
            catalog_snapshot=catalog_snapshot,
        )
        missing_capabilities = set(obligation["required_capability_ids"]) - capabilities
        if missing_checks or missing_evidence or missing_capabilities:
            raise ReactLoopError(
                "REACT_REQUIREMENT_COVERAGE_INCOMPLETE",
                f"{obligation_id!r} is missing checks={sorted(missing_checks)}, "
                f"evidence={sorted(missing_evidence)}, capabilities={sorted(missing_capabilities)}",
            )
        binding = obligation.get("acceptance_binding")
        if isinstance(binding, Mapping):
            matching_tests = [
                test_by_id[item]
                for item in plan_items
                if item in test_by_id
                and isinstance(test_by_id[item].get("target"), Mapping)
                and test_by_id[item]["target"].get("metric_id") == binding.get("metric_id")
                and test_by_id[item]["target"].get("load_case_id")
                == binding.get("test_load_case_id")
                and set(test_by_id[item].get("required_branches") or [])
                >= set(binding.get("required_branches") or [])
            ]
            if not matching_tests:
                raise ReactLoopError(
                    "REACT_REQUIREMENT_COVERAGE_ACCEPTANCE_MISMATCH",
                    f"{obligation_id!r} has no referenced test with its frozen metric, load case, and branches",
                )
        normalized.append(
            {
                "obligation_id": obligation_id,
                "plan_item_ids": plan_items,
                "model_element_ids": model_elements,
                "check_ids": check_ids,
                "evidence_roles": evidence_roles,
                "failure_diagnostic": _text(
                    raw.get("failure_diagnostic"),
                    label=f"{label}.failure_diagnostic",
                    limit=2_048,
                ),
            }
        )
    if seen != set(inventory_by_id):
        raise ReactLoopError(
            "REACT_REQUIREMENT_COVERAGE_INCOMPLETE",
            "obligation_coverage omits one or more frozen obligation IDs",
        )
    normalized.sort(key=lambda item: item["obligation_id"])
    return strict_json_copy(normalized, label="obligation coverage")


def requirement_check_catalog(
    inventory: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Describe only the Harness checks referenced by one frozen task."""

    entries = inventory.get("entries") if isinstance(inventory, Mapping) else None
    if not isinstance(entries, list):
        _fail(
            "REQUIREMENT_OBLIGATION_INVALID",
            "requirement obligation inventory entries are missing",
        )
    check_ids = sorted(
        {
            str(check_id)
            for entry in entries
            if isinstance(entry, Mapping)
            for check_id in entry.get("required_check_ids", [])
        }
    )
    catalog: list[dict[str, str]] = []
    metric_prefix = "validator.metric."
    for check_id in check_ids:
        mode = _registered_check_mode(check_id, label="required_check_ids")
        if check_id.startswith(metric_prefix):
            metric_id = check_id[len(metric_prefix) :]
            details = (
                f"The typed {metric_id!r} metric for its frozen load case and required branches.",
                "Every required branch produces authoritative evidence and the metric satisfies the frozen comparator, target, unit, and tolerance.",
                "Typed task acceptance evaluation and ideal-reference comparison when required.",
            )
        else:
            details = OBLIGATION_CHECK_DETAILS[check_id]
        catalog.append(
            {
                "check_id": check_id,
                "mode": mode,
                "examines": details[0],
                "pass_condition": details[1],
                "evidence": details[2],
            }
        )
    return strict_json_copy(catalog, label="requirement check catalog")


__all__ = [
    "OBLIGATION_CHECK_REGISTRY",
    "OBLIGATION_CHECK_DETAILS",
    "OBLIGATION_COVERAGE_SCHEMA",
    "OBLIGATION_INVENTORY_SCHEMA",
    "RequirementObligationError",
    "build_requirement_obligation_inventory",
    "requirement_check_catalog",
    "validate_obligation_coverage",
]
