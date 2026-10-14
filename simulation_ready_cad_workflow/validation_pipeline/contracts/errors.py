"""Shared contract error type used across the Loop/Harness boundary.

``ReactLoopError`` is the canonical typed error for every contract
violation that crosses a component boundary: invalid worker payloads,
broken planning contracts, failed submission bindings, and Harness
adapter misuse.  It lives in ``contracts`` so the Harness never has to
import Loop code to raise or catch it.
"""

from __future__ import annotations


class ReactLoopError(ValueError):
    """A turn request, response, contract document, or transition is invalid."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")
