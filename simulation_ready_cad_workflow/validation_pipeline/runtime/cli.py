from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .models import ManifestError, QueueManifest
from .plain_runner import PlainQueueRunError, PlainQueueRunner
from .system_probe import SystemProbe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simulation-test-queue",
        description="Run independent Harness test lanes with resource gates and fail-fast.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe")
    probe.add_argument("--json", action="store_true")
    for name in ("validate", "run"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", required=True)
        command.add_argument("--workspace-root", default=".")
        command.add_argument("--run-dir")
        command.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "probe":
            payload = SystemProbe().sample(1.0).to_dict()
            _print(payload, as_json=args.json)
            return 0
        manifest = QueueManifest.from_file(args.manifest)
        workspace = Path(args.workspace_root).resolve()
        run_dir = (
            Path(args.run_dir).resolve()
            if args.run_dir
            else workspace / ".runtime_runs" / manifest.run_name
        )
        runner = PlainQueueRunner(
            manifest,
            workspace_root=workspace,
            run_dir=run_dir,
        )
        if args.command == "validate":
            payload = {
                "status": "VALID",
                "run_name": manifest.run_name,
                "job_count": len(manifest.jobs),
                "max_parallel_jobs": manifest.queue_policy.max_parallel_jobs,
                "run_dir": run_dir.relative_to(workspace).as_posix(),
            }
        else:
            payload = runner.run()
        _print(payload, as_json=args.json)
        return 0 if payload["status"] in {"VALID", "SUCCEEDED"} else 3
    except (ManifestError, PlainQueueRunError, OSError, RuntimeError) as exc:
        print(f"simulation-test-queue: {exc}", file=sys.stderr)
        return 2


def _print(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    for key, value in payload.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    raise SystemExit(main())
