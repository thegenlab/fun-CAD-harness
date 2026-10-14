"""The frozen Harness result contract for ``<run>/run_status.json``.

``harness_result.v2.schema.json`` is the single source of truth for the
result envelope: its closed status vocabulary and per-status required
fields.  Every producer must publish through :func:`publish_run_status`,
which validates against that schema before anything reaches disk, so a
malformed result fails the producing branch instead of poisoning the run
directory for the Loop, dashboard, or later benchmark extraction.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .schema_check import SchemaCheckError, check_against_schema


HARNESS_RESULT_SCHEMA_ID = "simulation_ready_cad_harness_result_v2"
_SCHEMA_FILE = Path(__file__).resolve().parent / "harness_result.v2.schema.json"


class ContractViolation(ValueError):
    """A produced document does not satisfy its frozen contract."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@lru_cache(maxsize=1)
def harness_result_schema() -> dict[str, Any]:
    schema = json.loads(_SCHEMA_FILE.read_text(encoding="utf-8"))
    if schema.get("$id") != HARNESS_RESULT_SCHEMA_ID:
        raise ContractViolation(
            "CONTRACT_SCHEMA_INVALID",
            "harness result schema file does not carry the expected $id",
        )
    return schema


def harness_result_statuses() -> frozenset[str]:
    """The closed status vocabulary, read from the schema file."""

    return frozenset(
        harness_result_schema()["properties"]["status"]["enum"]
    )


def validate_harness_result(result: Mapping[str, Any]) -> None:
    try:
        check_against_schema(result, harness_result_schema())
    except SchemaCheckError as exc:
        raise ContractViolation(
            "HARNESS_RESULT_CONTRACT_VIOLATION",
            "; ".join(exc.errors),
        ) from exc


def publish_run_status(
    run_dir: str | Path,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate ``result`` and atomically write ``run_status.json``.

    This is the only supported way to publish a Harness result.  The write
    itself is delegated to the mechanism I/O helper so the on-disk format
    (encoding, atomic replace) stays identical everywhere.
    """

    from ..mechanism.io import write_json_atomic

    validate_harness_result(result)
    value = dict(result)
    write_json_atomic(Path(run_dir) / "run_status.json", value)
    # Every publication is journaled into the hash-chained execution trace;
    # a damaged chain must surface, not be papered over.
    from .traces import append_execution_trace

    append_execution_trace(
        run_dir,
        status=str(value.get("status")),
        last_completed_stage=value.get("last_completed_stage"),
        detail={
            "failure_reason": value.get("failure_reason"),
            "outcome_kind": value.get("outcome_kind"),
        },
    )
    return value
