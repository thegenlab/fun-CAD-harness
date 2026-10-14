"""Validator-owned benchmark TestSpec / Result contracts.

Catalog and obligation tests are sealed (``benchmark_validator``). Project
``authored_test_tasks`` live in the candidate trust domain and do not score
the mechanical verdict unless an evaluator explicitly opts into ``sealed``
authority for that run.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .harness_result import ContractViolation
from .schema_check import SchemaCheckError, check_against_schema

BENCHMARK_TEST_SPEC_SCHEMA_ID = "simulation_ready_cad_benchmark_test_spec_v1"
BENCHMARK_TEST_RESULT_SCHEMA_ID = "simulation_ready_cad_benchmark_test_result_v1"
SEALED_AUTHORITY = "benchmark_validator"
AUTHORED_AUTHORITY = "project_authored"
_DIGEST_LENGTH = 64

_SPEC_FILE = Path(__file__).resolve().parent / "benchmark_test_spec.v1.schema.json"
_RESULT_FILE = (
    Path(__file__).resolve().parent / "benchmark_test_result.v1.schema.json"
)


def test_layer_policy(authority: str) -> dict[str, Any]:
    """Closed mapping from test authority to scoring rights."""

    if authority in {SEALED_AUTHORITY, "sealed"}:
        normalized = SEALED_AUTHORITY if authority == SEALED_AUTHORITY else "sealed"
        return {
            "authority": normalized,
            "benchmark_fair": True,
            "scores_mechanical": True,
        }
    return {
        "authority": AUTHORED_AUTHORITY,
        "benchmark_fair": False,
        "scores_mechanical": False,
    }


@lru_cache(maxsize=1)
def benchmark_test_spec_schema() -> dict[str, Any]:
    schema = json.loads(_SPEC_FILE.read_text(encoding="utf-8"))
    if schema.get("$id") != BENCHMARK_TEST_SPEC_SCHEMA_ID:
        raise ContractViolation(
            "CONTRACT_SCHEMA_INVALID",
            "benchmark test spec schema file does not carry the expected $id",
        )
    return schema


@lru_cache(maxsize=1)
def benchmark_test_result_schema() -> dict[str, Any]:
    schema = json.loads(_RESULT_FILE.read_text(encoding="utf-8"))
    if schema.get("$id") != BENCHMARK_TEST_RESULT_SCHEMA_ID:
        raise ContractViolation(
            "CONTRACT_SCHEMA_INVALID",
            "benchmark test result schema file does not carry the expected $id",
        )
    return schema


def _require_digest(value: Any, *, label: str) -> None:
    if not isinstance(value, str) or len(value) != _DIGEST_LENGTH:
        raise ContractViolation(
            "BENCHMARK_TEST_CONTRACT_VIOLATION",
            f"{label} must be a 64-character digest",
        )
    try:
        int(value, 16)
    except ValueError as exc:
        raise ContractViolation(
            "BENCHMARK_TEST_CONTRACT_VIOLATION",
            f"{label} must be a hexadecimal digest",
        ) from exc


def validate_benchmark_test_spec(document: Mapping[str, Any]) -> None:
    try:
        check_against_schema(document, benchmark_test_spec_schema())
    except SchemaCheckError as exc:
        raise ContractViolation(
            "BENCHMARK_TEST_CONTRACT_VIOLATION",
            "; ".join(exc.errors),
        ) from exc
    bindings = document["bindings"]
    for field in (
        "requirements_sha256",
        "catalog_sha256",
        "validator_source_sha256",
    ):
        _require_digest(bindings[field], label=f"bindings.{field}")
    inventory = bindings.get("package_inventory_sha256")
    if inventory is not None:
        _require_digest(inventory, label="bindings.package_inventory_sha256")
    if document["evidence"]["missing_evidence"] != "FAIL":
        raise ContractViolation(
            "BENCHMARK_TEST_CONTRACT_VIOLATION",
            "missing evidence must fail closed",
        )


def validate_benchmark_test_result(document: Mapping[str, Any]) -> None:
    try:
        check_against_schema(document, benchmark_test_result_schema())
    except SchemaCheckError as exc:
        raise ContractViolation(
            "BENCHMARK_TEST_CONTRACT_VIOLATION",
            "; ".join(exc.errors),
        ) from exc
    for field in (
        "test_spec_sha256",
        "candidate_package_sha256",
        "validator_build_sha256",
    ):
        _require_digest(document[field], label=field)
    if document["status"] == "PASS" and not document.get("evidence"):
        raise ContractViolation(
            "BENCHMARK_TEST_CONTRACT_VIOLATION",
            "PASS requires evidence references",
        )
    roles = [
        str(item.get("role") or "")
        for item in document.get("evidence") or []
        if isinstance(item, Mapping)
    ]
    if "visualization" in roles and document.get("scores_mechanical") is True:
        raise ContractViolation(
            "BENCHMARK_TEST_CONTRACT_VIOLATION",
            "visualization evidence cannot score the mechanical verdict",
        )
