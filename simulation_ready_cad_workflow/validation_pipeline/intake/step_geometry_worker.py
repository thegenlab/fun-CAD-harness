from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .step_geometry import inspect_step_geometry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import one STEP file with OCCT and emit bounded JSON."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inspection = inspect_step_geometry(args.input)
    payload = json.dumps(
        inspection.to_dict(),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    temporary = args.output.with_name(
        f".{args.output.name}.{os.getpid()}.tmp"
    )
    encoded = (payload + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as stream:
            offset = 0
            while offset < len(encoded):
                written = stream.write(encoded[offset:])
                if not isinstance(written, int) or written <= 0:
                    raise OSError("worker result write made no forward progress")
                offset += written
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
