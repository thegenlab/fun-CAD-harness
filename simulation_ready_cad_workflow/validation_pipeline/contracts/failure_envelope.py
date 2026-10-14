"""Producer-owned failure facts carried across Harness boundaries.

The component that first knows why work stopped also knows which authority
can change that fact.  It records that authority here; outer queue and
Harness wrappers may add presentation status, but must preserve this envelope
unchanged.  Routing consumes ``owner`` for responsibility and an optional
producer-authored ``repair`` contract for change scope and DAG invalidation;
it never infers either from a display status or error-code prefix.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Mapping


FAILURE_ENVELOPE_SCHEMA = "simulation_ready_cad_failure_envelope_v1"
REPAIR_CONTRACT_SCHEMA = "simulation_ready_cad_repair_contract_v1"


class FailureOwner(str, Enum):
    DESIGN_PACKAGE = "design_package"
    WORKER_PLAN = "worker_plan"
    REVIEWER_MECHANISM_PLUGIN = "reviewer_mechanism_plugin"
    REVIEWER_TASK_PLUGIN = "reviewer_task_plugin"
    HARNESS_CODE = "harness_code"
    EXECUTION_RESOURCE = "execution_resource"
    UNCLASSIFIED = "unclassified"


class EvidenceValidity(str, Enum):
    VALID_MEASUREMENT = "VALID_MEASUREMENT"
    INVALID_MEASUREMENT = "INVALID_MEASUREMENT"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class RepairScope(str, Enum):
    """Smallest authority domain whose inputs may change for one repair."""

    RETRY_ONLY = "retry_only"
    DERIVED_ARTIFACT = "derived_artifact"
    PACKAGE_METADATA = "package_metadata"
    SIMULATION_SEMANTICS = "simulation_semantics"
    AUTHORITATIVE_GEOMETRY = "authoritative_geometry"
    PLAN_OR_CODE = "plan_or_code"


class FailureEnvelopeError(ValueError):
    """A producer supplied an incomplete or contradictory failure fact."""


_PLUGIN_REGISTRATION_OWNERS = {
    ("mechanism", "reviewer_mechanism_adapter"): (
        FailureOwner.REVIEWER_MECHANISM_PLUGIN
    ),
    ("mechanism", "project_design_package"): FailureOwner.DESIGN_PACKAGE,
    ("mechanism", "installed_harness_code"): FailureOwner.HARNESS_CODE,
    ("task", "reviewer_task_adapter"): FailureOwner.REVIEWER_TASK_PLUGIN,
    ("task", "project_design_package"): FailureOwner.DESIGN_PACKAGE,
    ("task", "installed_harness_code"): FailureOwner.HARNESS_CODE,
}


def mechanism_owner_for_registration_source(
    registration_source: str,
) -> FailureOwner:
    """Map one trusted mechanism-plugin registration to its repair authority."""

    return _PLUGIN_REGISTRATION_OWNERS.get(
        ("mechanism", str(registration_source or "")),
        FailureOwner.UNCLASSIFIED,
    )


def plugin_owner_for_registration(
    registration: Mapping[str, Any] | None,
    *,
    plugin_kind: str,
    expected_plugin_id: str | None = None,
) -> FailureOwner:
    """Resolve editable plugin authority from one trusted registration record.

    Runtime code must not infer authorship from a plugin-id namespace.  The
    registry record is the authority marker that survives dispatch and trace
    persistence.  A missing, mismatched, or unknown record deliberately stays
    unclassified so the bounded ownership-triage protocol can handle it.
    """

    if not isinstance(registration, Mapping):
        return FailureOwner.UNCLASSIFIED
    recorded_plugin_id = registration.get("plugin_id")
    if not isinstance(recorded_plugin_id, str) or not recorded_plugin_id.strip():
        return FailureOwner.UNCLASSIFIED
    if (
        expected_plugin_id is not None
        and recorded_plugin_id != str(expected_plugin_id).strip()
    ):
        return FailureOwner.UNCLASSIFIED
    source = str(registration.get("registration_source") or "")
    return _PLUGIN_REGISTRATION_OWNERS.get(
        (str(plugin_kind or ""), source),
        FailureOwner.UNCLASSIFIED,
    )


def _nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FailureEnvelopeError(f"{label} must be a non-empty string")
    return value.strip()


def _json_object(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise FailureEnvelopeError("failure details must be an object")
    return dict(value)


def _cause(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FailureEnvelopeError("each failure cause must be an object")
    result: dict[str, Any] = {
        "code": _nonempty(value.get("code"), label="failure cause code")
    }
    if value.get("message") is not None:
        result["message"] = _nonempty(
            value.get("message"), label="failure cause message"
        )
    details = _json_object(value.get("details"))
    if details is not None:
        result["details"] = details
    nested = value.get("causes")
    if nested is not None:
        if not isinstance(nested, list):
            raise FailureEnvelopeError("failure cause causes must be a list")
        result["causes"] = [_cause(item) for item in nested]
    return result


def _string_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise FailureEnvelopeError(f"{label} must be a list")
    result: list[str] = []
    for item in value:
        normalized = _nonempty(item, label=f"{label} item")
        if normalized not in result:
            result.append(normalized)
    return result


def make_repair_contract(
    scope: RepairScope | str,
    *,
    preserves: Iterable[str] = (),
    invalidates: Iterable[str] = (),
    resume_stage: str,
) -> dict[str, Any]:
    """Describe repair impact independently from who owns the failure."""

    try:
        selected_scope = RepairScope(scope)
    except (TypeError, ValueError) as exc:
        raise FailureEnvelopeError(f"unsupported repair scope {scope!r}") from exc
    return {
        "schema_version": REPAIR_CONTRACT_SCHEMA,
        "scope": selected_scope.value,
        "preserves": _string_list(list(preserves), label="repair preserves"),
        "invalidates": _string_list(
            list(invalidates), label="repair invalidates"
        ),
        "resume_stage": _nonempty(
            resume_stage, label="repair resume_stage"
        ),
    }


def validate_repair_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FailureEnvelopeError("failure repair contract must be an object")
    if value.get("schema_version") != REPAIR_CONTRACT_SCHEMA:
        raise FailureEnvelopeError("failure repair contract schema_version is invalid")
    return make_repair_contract(
        value.get("scope"),
        preserves=value.get("preserves") or (),
        invalidates=value.get("invalidates") or (),
        resume_stage=value.get("resume_stage"),
    )


def make_failure_envelope(
    code: str,
    owner: FailureOwner | str,
    *,
    source_component: str,
    message: str | None = None,
    evidence_validity: EvidenceValidity | str = EvidenceValidity.NOT_APPLICABLE,
    plugin_id: str | None = None,
    plugin_kind: str | None = None,
    registration_source: str | None = None,
    causes: Iterable[Mapping[str, Any]] = (),
    details: Mapping[str, Any] | None = None,
    repair: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create and validate one immutable-by-convention routing fact."""

    try:
        selected_owner = FailureOwner(owner)
    except (TypeError, ValueError) as exc:
        raise FailureEnvelopeError(f"unsupported failure owner {owner!r}") from exc
    try:
        selected_validity = EvidenceValidity(evidence_validity)
    except (TypeError, ValueError) as exc:
        raise FailureEnvelopeError(
            f"unsupported evidence validity {evidence_validity!r}"
        ) from exc
    source: dict[str, Any] = {
        "component": _nonempty(
            source_component, label="failure source component"
        )
    }
    if plugin_id is not None:
        source["plugin_id"] = _nonempty(plugin_id, label="failure plugin_id")
    if plugin_kind is not None:
        selected_kind = _nonempty(plugin_kind, label="failure plugin_kind")
        if selected_kind not in {"mechanism", "task"}:
            raise FailureEnvelopeError(
                "failure plugin_kind must be 'mechanism' or 'task'"
            )
        source["plugin_kind"] = selected_kind
    if registration_source is not None:
        source["registration_source"] = _nonempty(
            registration_source, label="failure registration_source"
        )
    expected_plugin_kind = {
        FailureOwner.REVIEWER_MECHANISM_PLUGIN: "mechanism",
        FailureOwner.REVIEWER_TASK_PLUGIN: "task",
    }.get(selected_owner)
    if expected_plugin_kind is not None and source.get("plugin_kind") != expected_plugin_kind:
        raise FailureEnvelopeError(
            f"{selected_owner.value} requires plugin_kind={expected_plugin_kind!r}"
        )

    result: dict[str, Any] = {
        "schema_version": FAILURE_ENVELOPE_SCHEMA,
        "code": _nonempty(code, label="failure code"),
        "owner": selected_owner.value,
        "source": source,
        "evidence_validity": selected_validity.value,
        "causes": [_cause(item) for item in causes],
    }
    if message is not None:
        result["message"] = _nonempty(message, label="failure message")
    normalized_details = _json_object(details)
    if normalized_details is not None:
        result["details"] = normalized_details
    if repair is not None:
        normalized_repair = validate_repair_contract(repair)
        design_scopes = {
            RepairScope.DERIVED_ARTIFACT.value,
            RepairScope.PACKAGE_METADATA.value,
            RepairScope.SIMULATION_SEMANTICS.value,
            RepairScope.AUTHORITATIVE_GEOMETRY.value,
        }
        if (
            normalized_repair["scope"] in design_scopes
            and selected_owner is not FailureOwner.DESIGN_PACKAGE
        ):
            raise FailureEnvelopeError(
                "design repair scopes require owner='design_package'"
            )
        result["repair"] = normalized_repair
    return result


def validate_failure_envelope(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached normalized envelope or raise on an invalid owner fact."""

    if not isinstance(value, Mapping):
        raise FailureEnvelopeError("failure envelope must be an object")
    if value.get("schema_version") != FAILURE_ENVELOPE_SCHEMA:
        raise FailureEnvelopeError("failure envelope schema_version is invalid")
    source = value.get("source")
    if not isinstance(source, Mapping):
        raise FailureEnvelopeError("failure envelope source must be an object")
    causes = value.get("causes")
    if not isinstance(causes, list):
        raise FailureEnvelopeError("failure envelope causes must be a list")
    return make_failure_envelope(
        value.get("code"),
        value.get("owner"),
        source_component=source.get("component"),
        message=value.get("message"),
        evidence_validity=value.get("evidence_validity"),
        plugin_id=source.get("plugin_id"),
        plugin_kind=source.get("plugin_kind"),
        registration_source=source.get("registration_source"),
        causes=causes,
        details=value.get("details"),
        repair=value.get("repair"),
    )


def _make_plugin_provenance_failure_envelope(
    code: str,
    *,
    owner: FailureOwner,
    registration: Mapping[str, Any] | None,
    plugin_id: str,
    plugin_kind: str,
    source_component: str,
    message: str | None = None,
    evidence_validity: EvidenceValidity | str = EvidenceValidity.NOT_APPLICABLE,
    causes: Iterable[Mapping[str, Any]] = (),
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    registration_source = (
        str(registration.get("registration_source") or "")
        if isinstance(registration, Mapping)
        else ""
    )
    return make_failure_envelope(
        code,
        owner,
        source_component=source_component,
        message=message,
        evidence_validity=evidence_validity,
        plugin_id=plugin_id,
        plugin_kind=plugin_kind,
        registration_source=(registration_source or None),
        causes=causes,
        details=details,
    )


def make_plugin_failure_envelope(
    code: str,
    *,
    registration: Mapping[str, Any] | None,
    plugin_id: str,
    plugin_kind: str,
    source_component: str,
    message: str | None = None,
    evidence_validity: EvidenceValidity | str = EvidenceValidity.NOT_APPLICABLE,
    causes: Iterable[Mapping[str, Any]] = (),
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assign a proven plugin contract/code failure to its registered author."""

    owner = plugin_owner_for_registration(
        registration,
        plugin_kind=plugin_kind,
        expected_plugin_id=plugin_id,
    )
    return _make_plugin_provenance_failure_envelope(
        code,
        owner=owner,
        registration=registration,
        plugin_id=plugin_id,
        plugin_kind=plugin_kind,
        source_component=source_component,
        message=message,
        evidence_validity=evidence_validity,
        causes=causes,
        details=details,
    )


def make_unclassified_plugin_failure_envelope(
    code: str,
    *,
    registration: Mapping[str, Any] | None,
    plugin_id: str,
    plugin_kind: str,
    source_component: str,
    message: str | None = None,
    evidence_validity: EvidenceValidity | str = EvidenceValidity.NOT_APPLICABLE,
    causes: Iterable[Mapping[str, Any]] = (),
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep ambiguous plugin input validation in bounded ownership triage."""

    return _make_plugin_provenance_failure_envelope(
        code,
        owner=FailureOwner.UNCLASSIFIED,
        registration=registration,
        plugin_id=plugin_id,
        plugin_kind=plugin_kind,
        source_component=source_component,
        message=message,
        evidence_validity=evidence_validity,
        causes=causes,
        details=details,
    )


def failure_envelope_from_exception(
    exc: BaseException,
    *,
    default_owner: FailureOwner,
    source_component: str,
    evidence_validity: EvidenceValidity = EvidenceValidity.NOT_APPLICABLE,
) -> dict[str, Any]:
    """Preserve a typed producer envelope, otherwise fail closed at this boundary."""

    carried = getattr(exc, "failure_envelope", None)
    if isinstance(carried, Mapping):
        return validate_failure_envelope(carried)
    code = str(getattr(exc, "code", type(exc).__name__))
    message = str(getattr(exc, "message", str(exc))) or code
    return make_failure_envelope(
        code,
        default_owner,
        source_component=source_component,
        message=message,
        evidence_validity=evidence_validity,
        details=(
            getattr(exc, "details", None)
            if isinstance(getattr(exc, "details", None), Mapping)
            else None
        ),
    )


def unclassified_failure_envelope(
    code: str,
    *,
    source_component: str,
    message: str | None = None,
    causes: Iterable[Mapping[str, Any]] = (),
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return make_failure_envelope(
        code,
        FailureOwner.UNCLASSIFIED,
        source_component=source_component,
        message=message,
        evidence_validity=EvidenceValidity.NOT_APPLICABLE,
        causes=causes,
        details=details,
    )


__all__ = [
    "EvidenceValidity",
    "FAILURE_ENVELOPE_SCHEMA",
    "REPAIR_CONTRACT_SCHEMA",
    "FailureEnvelopeError",
    "FailureOwner",
    "RepairScope",
    "failure_envelope_from_exception",
    "make_failure_envelope",
    "make_plugin_failure_envelope",
    "make_repair_contract",
    "make_unclassified_plugin_failure_envelope",
    "mechanism_owner_for_registration_source",
    "plugin_owner_for_registration",
    "unclassified_failure_envelope",
    "validate_failure_envelope",
    "validate_repair_contract",
]
