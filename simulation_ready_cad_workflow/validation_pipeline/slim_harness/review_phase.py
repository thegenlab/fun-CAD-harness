"""Explicit phase and finite evidence collection, without changing acceptance."""
from __future__ import annotations


def phase_prompt(criteria, *, final):
    ids=[row["id"] for row in criteria["criteria"]]
    if final:
        return (
            "CURRENT REVIEW PHASE: FINAL ACCEPTANCE, NOT PREFLIGHT.\n"
            "Physical results and original-geometry audits have already been supplied below. "
            "Return an observation for EVERY frozen criterion: "+", ".join(ids)+". "
            "Use actual physical measurements and audit evidence, not merely test coverage or a plan. "
            "Do not defer functional acceptance to a future review: this IS that review. "
            "ready requires every criterion met with observed facts and evidence paths. "
            "If a fact remains unobservable, keep it unobservable and return needs_evidence, never invent a PASS. "
            "Carry forward unchanged, supported appearance observations and their evidence even if the newest "
            "image does not repeat an earlier view. Reopen them only for a specific conflicting fact. "
            "Any preflight-only guidance below applies only to the earlier phase, not this request.\n\n")
    return (
        "CURRENT REVIEW PHASE: PREFLIGHT.\n"
        "Review source geometry, appearance and proposed test coverage. Future physical results are not yet "
        "required. ready here authorizes execution only, never final mechanical acceptance.\n\n")


def phase_schema(schema, criteria, *, final):
    ids=[row["id"] for row in criteria["criteria"]]
    observations=schema["properties"]["observations"]
    observations["items"]["properties"]["criterion_id"]={"type":"string","enum":ids}
    if final:
        observations["minItems"]=len(ids)
        observations["maxItems"]=len(ids)


def next_evidence_round(prior, scene_sha256, requests):
    """Stop repeated unresolved criteria after two newly rendered evidence rounds."""
    keys=sorted({(row["criterion_id"],row["action"]) for row in requests})
    old=prior.get("unresolved",[])
    previous=[tuple(row) for row in old]
    rounds=int(prior.get("rounds",0)) if prior.get("scene_sha256")==scene_sha256 and previous==keys else 0
    if not keys:
        raise ValueError("EVIDENCE_REQUEST_EMPTY: reviewer supplied no actionable evidence request")
    if rounds>=2:
        raise ValueError("VISUAL_EVIDENCE_STALLED: two new evidence rounds did not resolve the same criteria; "
                         "diagnose source visibility, rendering or missing measurement instead of repeating crops")
    return {"unresolved":[list(row) for row in keys],"rounds":rounds+1}
