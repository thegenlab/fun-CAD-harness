"""Bounded subprocess entry for one approved project mechanism adapter."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from ..contracts.failure_envelope import (
    EvidenceValidity,
    make_plugin_failure_envelope,
)
from .models import MechanismValidationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe or call one hash-bound project adapter in isolation."
    )
    parser.add_argument("action", choices=("probe", "call"))
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--relative", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--method", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .family_plugins import exec_project_adapter_in_process
    from .project_adapter_isolation import (
        decode_worker_value,
        encode_worker_value,
        project_adapter_descriptor,
    )

    try:
        plugin = exec_project_adapter_in_process(
            args.package, args.relative, args.sha256
        )
        if args.action == "probe":
            payload = {
                "ok": True,
                "descriptor": project_adapter_descriptor(plugin),
            }
        else:
            request = json.loads(sys.stdin.read() or "{}")
            method = str(args.method or request.get("method") or "")
            if not method or method.startswith("_"):
                raise MechanismValidationError(
                    "PROJECT_MECHANISM_ADAPTER_CALL_INVALID",
                    "project adapter method is missing or private",
                )
            positional = [
                decode_worker_value(item, plugin=plugin)
                for item in request.get("args") or ()
            ]
            named = {
                str(key): decode_worker_value(value, plugin=plugin)
                for key, value in dict(request.get("kwargs") or {}).items()
            }
            if method == "ideal_model":
                model = plugin.ideal_model(*positional, **named)
                encoded = (
                    None
                    if model is None
                    else {"__type__": "IsolatedIdealModel", "value": True}
                )
            elif method == "ideal_model_evaluate_samples":
                plan, family_spec, samples = positional
                model = plugin.ideal_model(plan, family_spec)
                if model is None:
                    raise MechanismValidationError(
                        "IDEAL_REFERENCE_UNSUPPORTED",
                        "project adapter returned no ideal model",
                    )
                evaluated, domain = model.evaluate_samples(plan, samples)
                encoded = encode_worker_value(
                    {"samples": evaluated, "domain": domain}
                )
            else:
                target = getattr(plugin, method, None)
                if not callable(target):
                    raise MechanismValidationError(
                        "PROJECT_MECHANISM_ADAPTER_CALL_INVALID",
                        f"project adapter has no callable {method!r}",
                    )
                encoded = encode_worker_value(target(*positional, **named))
            payload = {
                "ok": True,
                "result": encoded,
            }
    except MechanismValidationError as exc:
        payload = {
            "ok": False,
            "code": exc.code,
            "message": str(exc),
            "exception_kind": "typed_validation",
            "details": encode_worker_value(exc.details),
            "failure": exc.failure_envelope,
        }
    except Exception as exc:
        plugin_id = str(getattr(locals().get("plugin"), "plugin_id", ""))
        callback_failure = None
        if args.action == "call" and plugin_id:
            callback_failure = make_plugin_failure_envelope(
                "PROJECT_MECHANISM_ADAPTER_CALLBACK_FAILED",
                registration={
                    "plugin_id": plugin_id,
                    "registration_source": "project_design_package",
                },
                plugin_id=plugin_id,
                plugin_kind="mechanism",
                source_component="mechanism.project_adapter_worker",
                message=f"{type(exc).__name__}: {exc}",
                evidence_validity=EvidenceValidity.NOT_APPLICABLE,
                details={"callback": str(args.method or "")},
            )
        payload = {
            "ok": False,
            "code": (
                "PROJECT_MECHANISM_ADAPTER_CALLBACK_FAILED"
                if callback_failure is not None
                else "PROJECT_MECHANISM_ADAPTER_LOAD_FAILED"
            ),
            "message": f"{type(exc).__name__}: {exc}",
            "exception_kind": "unexpected",
            "details": {"callback": str(args.method or "")},
            "failure": callback_failure,
        }
    encoded = json.dumps(
        payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    )
    sys.stdout.write(encoded + "\n")
    return 0 if payload.get("ok") is True else 1


if __name__ == "__main__":
    os.environ["SIMREADY_PROJECT_ADAPTER_ISOLATE"] = "0"
    raise SystemExit(main())
