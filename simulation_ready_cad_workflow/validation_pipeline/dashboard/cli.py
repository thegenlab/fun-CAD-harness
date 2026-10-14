from __future__ import annotations

import argparse
import ipaddress
import json
from pathlib import Path

from .model import normalize_run
from .render import render_snapshot
from .server import serve
from .stage_manifest import parse_scope_json, register_workflow_stage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Closed-loop quest dashboard")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="serve a live dashboard")
    serve_parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    serve_parser.add_argument("--runs-root", type=Path, default=Path(".runtime_runs"))
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--base-path", default="", help="public URL prefix, e.g. /harness/dashboard")
    serve_parser.add_argument(
        "--enable-launch",
        action="store_true",
        help="serve the New task page and allow starting, stopping and resuming runs",
    )
    serve_parser.add_argument(
        "--allow-remote-launch",
        action="store_true",
        help="required with --enable-launch when --host is not a loopback address (e.g. 0.0.0.0 in Docker)",
    )
    serve_parser.add_argument(
        "--max-launches", type=int, default=4, help="runs started from the dashboard that may run at the same time (queued runs wait for all of them)"
    )

    export_parser = subparsers.add_parser(
        "export", help="export one self-contained dashboard snapshot"
    )
    export_parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    export_parser.add_argument("--run", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)

    stage_parser = subparsers.add_parser(
        "register-stage",
        help=(
            "register one evidence-bearing project-specific workflow stage; "
            "the Harness assigns EVIDENCE_COMPLETE during finalization"
        ),
    )
    stage_parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.cwd(),
    )
    stage_parser.add_argument("--run", type=Path, required=True)
    stage_parser.add_argument("--id", required=True, dest="stage_id")
    stage_parser.add_argument("--kind", required=True)
    stage_parser.add_argument("--label", required=True)
    stage_parser.add_argument(
        "--evidence",
        action="append",
        required=True,
        type=Path,
        help=(
            "workspace-relative or absolute evidence path inside the run; "
            "repeat for multiple files"
        ),
    )
    stage_parser.add_argument(
        "--depends-on",
        action="append",
        default=[],
        help="exact stage id dependency; repeat for multiple dependencies",
    )
    stage_parser.add_argument(
        "--scope-json",
        default="{}",
        help="bounded strict JSON object describing this stage instance",
    )
    stage_parser.add_argument(
        "--optional",
        action="store_true",
        help="record required=false (evidence is still mandatory)",
    )

    return parser


def _loopback(host: str) -> bool:
    try:
        return host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "serve" and args.enable_launch:
        if not _loopback(args.host) and not args.allow_remote_launch:
            parser.error(
                f"--enable-launch on {args.host!r} lets anyone who can reach this port and holds the "
                "printed link start agent runs; add --allow-remote-launch to confirm"
            )
        if args.max_launches < 1:
            parser.error("--max-launches must be at least 1")
    workspace = args.workspace_root.resolve()
    if args.command == "register-stage":
        run = args.run
        if not run.is_absolute():
            run = workspace / run
        registration = register_workflow_stage(
            workspace_root=workspace,
            run_dir=run,
            stage_id=args.stage_id,
            kind=args.kind,
            label=args.label,
            evidence_paths=args.evidence,
            required=not args.optional,
            depends_on=args.depends_on,
            scope=parse_scope_json(args.scope_json),
        )
        print(json.dumps(registration, ensure_ascii=False, indent=2))
        return 0
    if args.command == "serve":
        runs_root = args.runs_root
        if not runs_root.is_absolute():
            runs_root = workspace / runs_root
        serve(
            workspace_root=workspace,
            runs_root=runs_root,
            host=args.host,
            port=args.port,
            base_path=args.base_path,
            launch_enabled=args.enable_launch,
            launch_limit=args.max_launches,
        )
        return 0

    run = args.run
    if not run.is_absolute():
        run = workspace / run
    output = args.output
    if not output.is_absolute():
        output = workspace / output
    output = output.resolve()
    runtime_root = (workspace / ".runtime_runs").resolve()
    try:
        output.relative_to(runtime_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "dashboard export is a derivative and must stay outside "
            ".runtime_runs"
        )
    if output.is_symlink():
        raise ValueError("dashboard export output cannot be a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    data = normalize_run(run, workspace_root=workspace, mode="snapshot")
    output.write_text(render_snapshot(data), encoding="utf-8")
    print(output)
    return 0
