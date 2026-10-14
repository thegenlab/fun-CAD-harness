"""Harness change proposals: submit, independently review, decide, then land.

One hash-chained ledger per project
(``<project>/maintenance/harness_change_proposals.json``,
``harness_change_proposal_ledger_v1``) records three entry kinds:

* ``PROPOSAL`` — an Operator files a problem plus a proposed Harness
  change;
* ``REVIEW`` — an independent Harness-change AI records a hash-bound
  advisory recommendation; it never approves a change;
* ``DECISION`` — a human verdicts ``APPROVE`` / ``REVISE`` / ``REJECT``
  with a mandatory reason.

Every review and decision also appends automatically to the project's
maintenance trace (kinds ``proposal_review`` and ``proposal_decision``),
so the recommendation, final verdict, and reasons live in the same
tamper-evident record as every other Loop Loop intervention.

Landing is gated: :func:`land_approved_proposal` is the only sanctioned
wrapper for applying a proposed Harness change, and it fails closed for
anything that is not currently ``APPROVED`` — an undecided, revised, or
rejected proposal cannot land, so a rejected proposal leaves the Harness
byte-for-byte untouched.  Like the traces, a damaged ledger chain fails
closed instead of being read around.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Mapping

from .traces import (
    TraceContractError,
    _append_chained_entry,
    _load_chain,
    append_maintenance_record,
)

PROPOSAL_LEDGER_SCHEMA = "harness_change_proposal_ledger_v1"

PROPOSAL_VERDICTS = frozenset({"APPROVE", "REVISE", "REJECT"})
PROPOSAL_RECOMMENDATIONS = frozenset(
    {"RECOMMEND_APPROVE", "RECOMMEND_REVISE", "RECOMMEND_REJECT"}
)
STRICT_PROPOSAL_MODE = "STRICT_REVIEWED_V1"
HUMAN_DECISION_AUTHORITY = "HUMAN"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_STATUS_BY_VERDICT = {
    "APPROVE": "APPROVED",
    "REVISE": "REVISE_REQUESTED",
    "REJECT": "REJECTED",
}

_TERMINAL_STATUSES = frozenset({"APPROVED", "REJECTED"})


class ProposalError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def proposal_ledger_path(project_dir: str | Path) -> Path:
    return Path(project_dir) / "maintenance" / "harness_change_proposals.json"


def load_proposal_ledger(project_dir: str | Path) -> dict[str, Any]:
    """The verified ledger; fails closed on any chain damage."""

    return _load_chain(
        proposal_ledger_path(project_dir), PROPOSAL_LEDGER_SCHEMA
    )


def _require_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ProposalError(
            "PROPOSAL_FIELD_REQUIRED", f"{field} must be a non-empty string"
        )
    return text


def _require_sha256(value: Any, field: str) -> str:
    digest = _require_text(value, field)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ProposalError(
            "PROPOSAL_HASH_INVALID",
            f"{field} must be a lowercase SHA-256 digest",
        )
    return digest


def submit_proposal(
    project_dir: str | Path,
    *,
    title: str,
    problem: str,
    proposed_change: str,
    submitted_by: str,
    reproduction_sha256: str | None = None,
    proposed_diff_sha256: str | None = None,
    reproduction_ref: str | None = None,
    proposed_diff_ref: str | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """File one issue + Harness change proposal on the project ledger.

    Old callers remain legacy-compatible.  The standard strict path sets
    ``strict=True`` and supplies the exact reproduction and proposed-diff
    hashes; it then requires an independent advisory review and a human
    decision before landing.
    """

    ledger = load_proposal_ledger(project_dir)
    path = proposal_ledger_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    proposal_core: dict[str, Any] = {
        "entry_kind": "PROPOSAL",
        "title": _require_text(title, "title"),
        "problem": _require_text(problem, "problem"),
        "proposed_change": _require_text(
            proposed_change, "proposed_change"
        ),
        "submitted_by": _require_text(submitted_by, "submitted_by"),
    }
    if strict:
        proposal_core.update(
            {
                "proposal_mode": STRICT_PROPOSAL_MODE,
                "reproduction_sha256": _require_sha256(
                    reproduction_sha256, "reproduction_sha256"
                ),
                "reproduction_ref": _require_text(
                    reproduction_ref, "reproduction_ref"
                ),
                "proposed_diff_sha256": _require_sha256(
                    proposed_diff_sha256, "proposed_diff_sha256"
                ),
                "proposed_diff_ref": _require_text(
                    proposed_diff_ref, "proposed_diff_ref"
                ),
            }
        )
    elif any(
        value is not None
        for value in (
            reproduction_sha256,
            proposed_diff_sha256,
            reproduction_ref,
            proposed_diff_ref,
        )
    ):
        raise ProposalError(
            "PROPOSAL_STRICT_MODE_REQUIRED",
            "artifact refs and hashes may only be supplied with strict=True",
        )
    if strict:
        for existing in ledger["entries"]:
            if existing.get("entry_kind") != "PROPOSAL":
                continue
            if all(
                existing.get(key) == value
                for key, value in proposal_core.items()
            ):
                return dict(existing)
    proposal_count = sum(
        1
        for entry in ledger["entries"]
        if entry.get("entry_kind") == "PROPOSAL"
    )
    payload = {
        **proposal_core,
        "proposal_id": f"p{proposal_count + 1:04d}",
    }
    return _append_chained_entry(
        path,
        PROPOSAL_LEDGER_SCHEMA,
        payload,
    )


def _states(ledger: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for entry in ledger["entries"]:
        kind = entry.get("entry_kind")
        proposal_id = str(entry.get("proposal_id"))
        if kind == "PROPOSAL":
            strict = entry.get("proposal_mode") == STRICT_PROPOSAL_MODE
            states[proposal_id] = {
                "proposal_id": proposal_id,
                "title": entry.get("title"),
                "problem": entry.get("problem"),
                "proposed_change": entry.get("proposed_change"),
                "submitted_by": entry.get("submitted_by"),
                "submitted_at": entry.get("created_at"),
                "proposal_entry_sha256": entry.get("entry_sha256"),
                "proposal_mode": (
                    STRICT_PROPOSAL_MODE if strict else "LEGACY_V1"
                ),
                "strict_review_required": strict,
                "reproduction_sha256": entry.get("reproduction_sha256"),
                "reproduction_ref": entry.get("reproduction_ref"),
                "proposed_diff_sha256": entry.get(
                    "proposed_diff_sha256"
                ),
                "proposed_diff_ref": entry.get("proposed_diff_ref"),
                "status": "REVIEW_REQUIRED" if strict else "SUBMITTED",
                "reviews": [],
                "decisions": [],
                "review_current": False,
                "current_review": None,
                "_latest_review_sequence": 0,
                "_latest_decision_sequence": 0,
            }
        elif kind == "REVIEW" and proposal_id in states:
            state = states[proposal_id]
            review = {
                "recommendation": entry.get("recommendation"),
                "reason": entry.get("reason"),
                "reviewed_by": entry.get("reviewed_by"),
                "reviewed_at": entry.get("created_at"),
                "proposal_entry_sha256": entry.get(
                    "proposal_entry_sha256"
                ),
                "reproduction_sha256": entry.get("reproduction_sha256"),
                "proposed_diff_sha256": entry.get(
                    "proposed_diff_sha256"
                ),
                "review_entry_sha256": entry.get("entry_sha256"),
            }
            state["reviews"].append(review)
            state["current_review"] = review
            state["_latest_review_sequence"] = int(
                entry.get("sequence") or 0
            )
            if state["status"] not in _TERMINAL_STATUSES:
                state["status"] = "HUMAN_DECISION_REQUIRED"
        elif kind == "DECISION" and proposal_id in states:
            state = states[proposal_id]
            state["decisions"].append(
                {
                    "verdict": entry.get("verdict"),
                    "reason": entry.get("reason"),
                    "decided_by": entry.get("decided_by"),
                    "decided_at": entry.get("created_at"),
                    "decision_authority": entry.get(
                        "decision_authority"
                    ),
                    "proposal_entry_sha256": entry.get(
                        "proposal_entry_sha256"
                    ),
                    "review_entry_sha256": entry.get(
                        "review_entry_sha256"
                    ),
                    "reproduction_sha256": entry.get(
                        "reproduction_sha256"
                    ),
                    "proposed_diff_sha256": entry.get(
                        "proposed_diff_sha256"
                    ),
                    "decision_entry_sha256": entry.get("entry_sha256"),
                }
            )
            state["_latest_decision_sequence"] = int(
                entry.get("sequence") or 0
            )
            state["status"] = _STATUS_BY_VERDICT.get(
                str(entry.get("verdict")), state["status"]
            )
    for state in states.values():
        state["review_current"] = (
            state["_latest_review_sequence"]
            > state["_latest_decision_sequence"]
        )
        if not state["review_current"]:
            state["current_review"] = None
        del state["_latest_review_sequence"]
        del state["_latest_decision_sequence"]
    return states


def proposal_state(
    project_dir: str | Path, proposal_id: str
) -> dict[str, Any]:
    states = _states(load_proposal_ledger(project_dir))
    state = states.get(proposal_id)
    if state is None:
        raise ProposalError(
            "PROPOSAL_UNKNOWN",
            f"proposal {proposal_id!r} is not on this project's ledger",
        )
    return state


def _assert_exact_binding(
    state: Mapping[str, Any],
    *,
    proposal_entry_sha256: Any,
    reproduction_sha256: Any,
    proposed_diff_sha256: Any,
) -> tuple[str, str, str]:
    supplied = (
        _require_sha256(proposal_entry_sha256, "proposal_entry_sha256"),
        _require_sha256(reproduction_sha256, "reproduction_sha256"),
        _require_sha256(proposed_diff_sha256, "proposed_diff_sha256"),
    )
    expected = (
        state.get("proposal_entry_sha256"),
        state.get("reproduction_sha256"),
        state.get("proposed_diff_sha256"),
    )
    if supplied != expected:
        raise ProposalError(
            "PROPOSAL_BINDING_MISMATCH",
            "the proposal, reproduction, or proposed-diff hash changed",
        )
    return supplied


def review_proposal(
    project_dir: str | Path,
    *,
    proposal_id: str,
    recommendation: str,
    reason: str,
    reviewed_by: str,
    proposal_entry_sha256: str,
    reproduction_sha256: str,
    proposed_diff_sha256: str,
) -> dict[str, Any]:
    """Record an independent, hash-bound AI recommendation.

    A ``REVIEW`` is advisory.  Even ``RECOMMEND_APPROVE`` leaves the
    proposal at ``HUMAN_DECISION_REQUIRED``; only a later human
    ``DECISION`` can change it to ``APPROVED``.
    """

    if recommendation not in PROPOSAL_RECOMMENDATIONS:
        raise ProposalError(
            "PROPOSAL_RECOMMENDATION_UNKNOWN",
            f"{recommendation!r} is not one of "
            f"{sorted(PROPOSAL_RECOMMENDATIONS)}",
        )
    reason_text = _require_text(reason, "reason")
    reviewed_by_text = _require_text(reviewed_by, "reviewed_by")
    state = proposal_state(project_dir, proposal_id)
    if not state["strict_review_required"]:
        raise ProposalError(
            "PROPOSAL_LEGACY_REVIEW_UNSUPPORTED",
            "legacy proposals retain their original direct-decision path",
        )
    if state["status"] in _TERMINAL_STATUSES:
        raise ProposalError(
            "PROPOSAL_ALREADY_DECIDED",
            f"proposal {proposal_id!r} is already {state['status']}",
        )
    if reviewed_by_text == state["submitted_by"]:
        raise ProposalError(
            "PROPOSAL_REVIEWER_NOT_INDEPENDENT",
            "the Harness Operator who submitted the proposal cannot review it",
        )
    proposal_hash, reproduction_hash, diff_hash = _assert_exact_binding(
        state,
        proposal_entry_sha256=proposal_entry_sha256,
        reproduction_sha256=reproduction_sha256,
        proposed_diff_sha256=proposed_diff_sha256,
    )
    entry = _append_chained_entry(
        proposal_ledger_path(project_dir),
        PROPOSAL_LEDGER_SCHEMA,
        {
            "entry_kind": "REVIEW",
            "proposal_id": proposal_id,
            "recommendation": recommendation,
            "reason": reason_text,
            "reviewed_by": reviewed_by_text,
            "proposal_entry_sha256": proposal_hash,
            "reproduction_sha256": reproduction_hash,
            "proposed_diff_sha256": diff_hash,
        },
    )
    append_maintenance_record(
        project_dir,
        actor=reviewed_by_text,
        intervention_kind="proposal_review",
        reason=f"{recommendation} {proposal_id}: {reason_text}",
        worker_visible_change="none",
        evidence={
            "proposal_id": proposal_id,
            "recommendation": recommendation,
            "proposal_entry_sha256": proposal_hash,
            "reproduction_sha256": reproduction_hash,
            "proposed_diff_sha256": diff_hash,
            "ledger_entry_sha256": entry["entry_sha256"],
        },
    )
    return entry


def decide_proposal(
    project_dir: str | Path,
    *,
    proposal_id: str,
    verdict: str,
    reason: str,
    decided_by: str,
    decision_authority: str | None = None,
    human_authorized: bool = False,
) -> dict[str, Any]:
    """Record one final verdict; strict proposals are HUMAN-only."""

    if verdict not in PROPOSAL_VERDICTS:
        raise ProposalError(
            "PROPOSAL_VERDICT_UNKNOWN",
            f"{verdict!r} is not one of {sorted(PROPOSAL_VERDICTS)}",
        )
    reason_text = _require_text(reason, "reason")
    decided_by_text = _require_text(decided_by, "decided_by")
    state = proposal_state(project_dir, proposal_id)
    if state["status"] in _TERMINAL_STATUSES:
        raise ProposalError(
            "PROPOSAL_ALREADY_DECIDED",
            f"proposal {proposal_id!r} is already {state['status']}",
        )
    payload: dict[str, Any] = {
        "entry_kind": "DECISION",
        "proposal_id": proposal_id,
        "verdict": verdict,
        "reason": reason_text,
        "decided_by": decided_by_text,
    }
    if state["strict_review_required"]:
        if (
            decision_authority != HUMAN_DECISION_AUTHORITY
            or human_authorized is not True
        ):
            raise ProposalError(
                "PROPOSAL_HUMAN_DECISION_REQUIRED",
                "strict proposals require an authenticated HUMAN final decision",
            )
        review = state.get("current_review")
        if not state.get("review_current") or not isinstance(review, dict):
            raise ProposalError(
                "PROPOSAL_REVIEW_REQUIRED",
                "a current independent AI review is required before the "
                "human decision",
            )
        if decided_by_text == review.get("reviewed_by"):
            raise ProposalError(
                "PROPOSAL_HUMAN_NOT_INDEPENDENT",
                "the AI reviewer cannot also record the human decision",
            )
        payload.update(
            {
                "decision_authority": HUMAN_DECISION_AUTHORITY,
                "proposal_entry_sha256": state[
                    "proposal_entry_sha256"
                ],
                "review_entry_sha256": review[
                    "review_entry_sha256"
                ],
                "reproduction_sha256": state["reproduction_sha256"],
                "proposed_diff_sha256": state[
                    "proposed_diff_sha256"
                ],
            }
        )
    entry = _append_chained_entry(
        proposal_ledger_path(project_dir),
        PROPOSAL_LEDGER_SCHEMA,
        payload,
    )
    trace_evidence = {
        "proposal_id": proposal_id,
        "verdict": verdict,
        "proposal_title": state["title"],
        "ledger_entry_sha256": entry["entry_sha256"],
    }
    if state["strict_review_required"]:
        trace_evidence.update(
            {
                "decision_authority": HUMAN_DECISION_AUTHORITY,
                "proposal_entry_sha256": payload[
                    "proposal_entry_sha256"
                ],
                "review_entry_sha256": payload["review_entry_sha256"],
                "reproduction_sha256": payload["reproduction_sha256"],
                "proposed_diff_sha256": payload[
                    "proposed_diff_sha256"
                ],
            }
        )
    append_maintenance_record(
        project_dir,
        actor=decided_by_text,
        intervention_kind="proposal_decision",
        reason=f"{verdict} {proposal_id}: {reason_text}",
        worker_visible_change="none",
        evidence=trace_evidence,
    )
    return entry


def assert_proposal_approved(
    project_dir: str | Path,
    proposal_id: str,
    *,
    reproduction_sha256: str | None = None,
    proposed_diff_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless approval and all strict bindings still match."""

    state = proposal_state(project_dir, proposal_id)
    if state["status"] != "APPROVED":
        raise ProposalError(
            "PROPOSAL_NOT_APPROVED",
            f"proposal {proposal_id!r} is {state['status']}; only an "
            "APPROVED proposal may land on the Harness",
        )
    if state["strict_review_required"]:
        decision = state["decisions"][-1]
        if (
            decision.get("decision_authority")
            != HUMAN_DECISION_AUTHORITY
        ):
            raise ProposalError(
                "PROPOSAL_HUMAN_APPROVAL_REQUIRED",
                "the final APPROVE is not a HUMAN decision",
            )
        review_sha = decision.get("review_entry_sha256")
        review = next(
            (
                item
                for item in state["reviews"]
                if item.get("review_entry_sha256") == review_sha
            ),
            None,
        )
        exact = (
            state["proposal_entry_sha256"],
            state["reproduction_sha256"],
            state["proposed_diff_sha256"],
        )
        decision_binding = (
            decision.get("proposal_entry_sha256"),
            decision.get("reproduction_sha256"),
            decision.get("proposed_diff_sha256"),
        )
        review_binding = (
            review.get("proposal_entry_sha256") if review else None,
            review.get("reproduction_sha256") if review else None,
            review.get("proposed_diff_sha256") if review else None,
        )
        if decision_binding != exact or review_binding != exact:
            raise ProposalError(
                "PROPOSAL_BINDING_MISMATCH",
                "review or human approval is not bound to the current exact "
                "proposal artifacts",
            )
        supplied_reproduction = _require_sha256(
            reproduction_sha256, "reproduction_sha256"
        )
        supplied_diff = _require_sha256(
            proposed_diff_sha256, "proposed_diff_sha256"
        )
        if supplied_reproduction != exact[1] or supplied_diff != exact[2]:
            raise ProposalError(
                "PROPOSAL_BINDING_MISMATCH",
                "the reproduction or proposed diff changed before landing",
            )
    return state


def land_approved_proposal(
    project_dir: str | Path,
    *,
    proposal_id: str,
    actor: str,
    apply: Callable[[], Any],
    reproduction_sha256: str | None = None,
    proposed_diff_sha256: str | None = None,
    landing_evidence_from_result: Callable[[Any], Mapping[str, Any]]
    | None = None,
) -> dict[str, Any]:
    """Apply one approved Harness change and record the landing.

    This is the only sanctioned wrapper for landing a proposal: the gate
    runs before ``apply`` so an unapproved or rejected proposal never
    executes any change, and the landing itself enters the maintenance
    trace with the proposal identity.
    """

    state = assert_proposal_approved(
        project_dir,
        proposal_id,
        reproduction_sha256=reproduction_sha256,
        proposed_diff_sha256=proposed_diff_sha256,
    )
    result = apply()
    landing_evidence = {
        "proposal_id": proposal_id,
        "proposal_entry_sha256": state["proposal_entry_sha256"],
    }
    if state["strict_review_required"]:
        decision = state["decisions"][-1]
        landing_evidence.update(
            {
                "decision_entry_sha256": decision[
                    "decision_entry_sha256"
                ],
                "review_entry_sha256": decision["review_entry_sha256"],
                "reproduction_sha256": state["reproduction_sha256"],
                "proposed_diff_sha256": state[
                    "proposed_diff_sha256"
                ],
            }
        )
    if landing_evidence_from_result is not None:
        landing_evidence = {
            **dict(landing_evidence_from_result(result)),
            **landing_evidence,
        }
    record = append_maintenance_record(
        project_dir,
        actor=_require_text(actor, "actor"),
        intervention_kind="proposal_landing",
        reason=(
            f"landed {proposal_id}: {state['proposed_change']}"
        ),
        worker_visible_change="none",
        evidence=landing_evidence,
    )
    return {"landing_record": record, "apply_result": result}


def proposals_view(project_dir: str | Path) -> dict[str, Any]:
    """Read-only dashboard section; chain damage reported, not hidden."""

    try:
        ledger = load_proposal_ledger(project_dir)
    except TraceContractError as error:
        return {
            "available": False,
            "ledger_intact": False,
            "error": str(error),
            "proposals": [],
        }
    states = list(_states(ledger).values())
    return {
        "available": True,
        "ledger_intact": True,
        "error": None,
        "proposals": states,
    }
