"""Compatibility notice for the removed pre-ReAct mechanism entry point."""

from __future__ import annotations

import json
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    print(
        json.dumps(
            {
                "status": "REMOVED",
                "message": (
                    "The old mechanism run/continue/capability entry point is "
                    "not part of the current Loop. Use "
                    "validation_pipeline.react_loop start-task/run-harness."
                ),
            },
            ensure_ascii=False,
        )
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
