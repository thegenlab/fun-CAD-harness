from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .api import validate_design_package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="design-package-intake",
        description=(
            "Validate an AI-generated simulation-ready CAD design package."
        ),
    )
    parser.add_argument("--package", required=True, help="Package directory")
    parser.add_argument(
        "--requirements",
        required=True,
        help="Immutable external requirements JSON",
    )
    parser.add_argument(
        "--output",
        help="Optional path for the complete JSON intake report",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Console output format (default: text)",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Return exit code 1 for ACCEPTED_WITH_WARNINGS",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    package_dir = Path(args.package).expanduser().resolve()
    requirements_path = Path(args.requirements).expanduser().resolve()
    if args.output:
        output = Path(args.output).expanduser().resolve()
        if output == requirements_path:
            print(
                "design-package-intake: --output must not overwrite "
                "the immutable requirements file",
                file=sys.stderr,
            )
            return 2
        try:
            output.relative_to(package_dir)
        except ValueError:
            pass
        else:
            print(
                "design-package-intake: --output must be outside "
                "the package directory",
                file=sys.stderr,
            )
            return 2
    report = validate_design_package(package_dir, requirements_path)
    if args.output:
        try:
            report.write_json(args.output)
        except OSError as exc:
            print(
                f"design-package-intake: could not write report: {exc}",
                file=sys.stderr,
            )
            return 2
    print(report.to_json() if args.format == "json" else report.render_text())
    return report.exit_code(fail_on_warning=args.fail_on_warning)


if __name__ == "__main__":
    raise SystemExit(main())
