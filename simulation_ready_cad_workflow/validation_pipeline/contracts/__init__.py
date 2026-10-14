"""Frozen I/O contracts between the Harness, the Loop, and consumers.

Each contract is a versioned JSON Schema file in this package plus a
validating publisher.  Schemas here define the interface boundary: the
Loop, dashboard, and benchmark tooling may rely on exactly what these
schemas guarantee and nothing else.
"""

from .errors import ReactLoopError
from .failure_envelope import (
    EvidenceValidity,
    FAILURE_ENVELOPE_SCHEMA,
    REPAIR_CONTRACT_SCHEMA,
    FailureEnvelopeError,
    FailureOwner,
    RepairScope,
    make_failure_envelope,
    make_plugin_failure_envelope,
    make_repair_contract,
    make_unclassified_plugin_failure_envelope,
    mechanism_owner_for_registration_source,
    plugin_owner_for_registration,
    unclassified_failure_envelope,
    validate_failure_envelope,
    validate_repair_contract,
)
from .harness_adapter import HarnessAdapter, HarnessOutcome, OutcomeKind
from .harness_result import (
    HARNESS_RESULT_SCHEMA_ID,
    ContractViolation,
    harness_result_schema,
    harness_result_statuses,
    publish_run_status,
    validate_harness_result,
)
from .schema_check import SchemaCheckError, check_against_schema

__all__ = [
    "HARNESS_RESULT_SCHEMA_ID",
    "ContractViolation",
    "EvidenceValidity",
    "FAILURE_ENVELOPE_SCHEMA",
    "REPAIR_CONTRACT_SCHEMA",
    "FailureEnvelopeError",
    "FailureOwner",
    "RepairScope",
    "HarnessAdapter",
    "HarnessOutcome",
    "OutcomeKind",
    "ReactLoopError",
    "SchemaCheckError",
    "check_against_schema",
    "harness_result_schema",
    "harness_result_statuses",
    "make_failure_envelope",
    "make_plugin_failure_envelope",
    "make_repair_contract",
    "make_unclassified_plugin_failure_envelope",
    "mechanism_owner_for_registration_source",
    "plugin_owner_for_registration",
    "publish_run_status",
    "unclassified_failure_envelope",
    "validate_failure_envelope",
    "validate_repair_contract",
    "validate_harness_result",
]
