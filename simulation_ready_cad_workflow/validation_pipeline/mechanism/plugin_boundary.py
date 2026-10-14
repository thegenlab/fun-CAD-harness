"""One ownership boundary for project, reviewer, and installed plugins.

Plugin provenance answers who can edit a broken callback implementation.  It
does not prove that every validation error raised inside that callback is a
code defect: a plugin may correctly reject bad plan or CAD input.  This module
keeps that distinction in one place so discovery, dispatch, lane execution,
and finalization cannot drift into different routing rules.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, TypeVar

from ..contracts.failure_envelope import (
    EvidenceValidity,
    make_plugin_failure_envelope,
    make_unclassified_plugin_failure_envelope,
)
from .models import MechanismValidationError


_T = TypeVar("_T")


def plugin_boundary_error(
    exc: BaseException,
    *,
    registration: Mapping[str, Any] | None,
    plugin_id: str,
    plugin_kind: str,
    callback: str,
    source_component: str,
    code: str | None = None,
    message: str | None = None,
    details: Mapping[str, Any] | None = None,
    contract_violation: bool = False,
    evidence_validity: EvidenceValidity = EvidenceValidity.NOT_APPLICABLE,
) -> MechanismValidationError:
    """Attach author provenance without mistaking detection site for cause.

    An existing producer envelope is authoritative and passes through.  An
    unexpected exception or a return-contract violation belongs to the code's
    registered author.  A typed but owner-less validation rejection remains
    unclassified, because it may describe bad input rather than broken plugin
    code; the Loop's bounded ownership triage resolves that case.
    """

    if (
        isinstance(exc, MechanismValidationError)
        and exc.failure_envelope is not None
    ):
        return exc

    original_code = str(getattr(exc, "code", type(exc).__name__))
    original_message = str(getattr(exc, "message", str(exc)))
    selected_code = str(code or original_code)
    selected_message = str(message or original_message or selected_code)
    failure_details = dict(
        getattr(exc, "details", {})
        if isinstance(getattr(exc, "details", None), Mapping)
        else {}
    )
    if isinstance(details, Mapping):
        failure_details.update(details)
    failure_details.setdefault("plugin_id", plugin_id)
    failure_details.setdefault("callback", callback)

    plugin_owned = contract_violation or not isinstance(
        exc, MechanismValidationError
    )
    envelope_factory = (
        make_plugin_failure_envelope
        if plugin_owned
        else make_unclassified_plugin_failure_envelope
    )
    cause = {"code": original_code}
    if original_message.strip():
        cause["message"] = original_message
    causes = (
        [cause]
        if selected_code != original_code or selected_message != original_message
        else []
    )
    failure = envelope_factory(
        selected_code,
        registration=registration,
        plugin_id=plugin_id,
        plugin_kind=plugin_kind,
        source_component=source_component,
        message=selected_message,
        evidence_validity=(
            evidence_validity
            if plugin_owned
            else EvidenceValidity.NOT_APPLICABLE
        ),
        causes=causes,
        details=failure_details,
    )
    error_type = (
        type(exc)
        if isinstance(exc, MechanismValidationError)
        else MechanismValidationError
    )
    return error_type(
        selected_code,
        selected_message,
        details=failure_details,
        failure=failure,
    )


def invoke_plugin_callback(
    operation: Callable[[], _T],
    *,
    registration: Mapping[str, Any] | None,
    plugin_id: str,
    plugin_kind: str,
    callback: str,
    source_component: str,
    unexpected_code: str,
    details: Mapping[str, Any] | None = None,
    evidence_validity: EvidenceValidity = EvidenceValidity.NOT_APPLICABLE,
) -> _T:
    """Invoke one plugin hook and preserve or create exactly one owner fact."""

    try:
        return operation()
    except MechanismValidationError as exc:
        boundary_error = plugin_boundary_error(
            exc,
            registration=registration,
            plugin_id=plugin_id,
            plugin_kind=plugin_kind,
            callback=callback,
            source_component=source_component,
            details=details,
        )
        if boundary_error is exc:
            raise
        raise boundary_error from exc
    except Exception as exc:
        message = f"{plugin_kind} plugin {callback}() failed: {exc}"
        raise plugin_boundary_error(
            exc,
            registration=registration,
            plugin_id=plugin_id,
            plugin_kind=plugin_kind,
            callback=callback,
            source_component=source_component,
            code=unexpected_code,
            message=message,
            details=details,
            evidence_validity=evidence_validity,
        ) from exc


__all__ = ["invoke_plugin_callback", "plugin_boundary_error"]
