from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


MAX_EVIDENCE_TEXT = 4_096
MAX_EVIDENCE_ITEMS = 100
MAX_EVIDENCE_DEPTH = 6
MAX_REPORTED_ISSUES = 10_000
MAX_REPORTED_ISSUE_BYTES = 64 * 1024 * 1024


def _bounded_text(value: str, limit: int = MAX_EVIDENCE_TEXT) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return value[:limit] + f"... <{omitted} characters omitted>"


def _bounded_evidence(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _bounded_text(value)
    if isinstance(value, Path):
        return _bounded_text(str(value))
    if isinstance(value, int):
        if value.bit_length() > 4_096:
            return f"<integer with {value.bit_length()} bits>"
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, bytes):
        prefix = value[:MAX_EVIDENCE_TEXT].hex()
        if len(value) > MAX_EVIDENCE_TEXT:
            prefix += f"... <{len(value) - MAX_EVIDENCE_TEXT} bytes omitted>"
        return prefix
    if depth >= MAX_EVIDENCE_DEPTH:
        return f"<{type(value).__name__} omitted at evidence depth limit>"
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        total_items = len(value)
        for index, (key, child) in enumerate(value.items()):
            if index >= MAX_EVIDENCE_ITEMS:
                break
            try:
                label = _bounded_text(str(key), 256)
            except Exception:
                label = f"<{type(key).__name__} key>"
            if label in result:
                label = f"{label} <duplicate key representation>"
            result[label] = _bounded_evidence(child, depth=depth + 1)
        if total_items > MAX_EVIDENCE_ITEMS:
            result["_truncated_items"] = total_items - MAX_EVIDENCE_ITEMS
        return result
    if isinstance(value, (list, tuple)):
        result = [
            _bounded_evidence(item, depth=depth + 1)
            for item in value[:MAX_EVIDENCE_ITEMS]
        ]
        if len(value) > MAX_EVIDENCE_ITEMS:
            result.append(
                {"_truncated_items": len(value) - MAX_EVIDENCE_ITEMS}
            )
        return result
    if isinstance(value, (set, frozenset)):
        ordered = sorted(value, key=lambda item: repr(item))
        return _bounded_evidence(ordered, depth=depth)
    try:
        rendered = repr(value)
    except Exception:
        rendered = f"<unrepresentable {type(value).__name__}>"
    return _bounded_text(rendered)


class Severity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class IntakeStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    ACCEPTED_WITH_WARNINGS = "ACCEPTED_WITH_WARNINGS"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class IntakeIssue:
    code: str
    severity: Severity
    message: str
    file: str | None = None
    json_path: str | None = None
    expected: Any = None
    actual: Any = None
    hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["severity"] = self.severity.value
        return {key: value for key, value in payload.items() if value is not None}


@dataclass
class IntakeReport:
    package_dir: str
    requirements_file: str | None
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    issues: list[IntakeIssue] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    capabilities: dict[str, Any] = field(default_factory=dict)
    package_summary: dict[str, Any] = field(default_factory=dict)
    _suppressed_issue_counts: dict[Severity, int] = field(
        default_factory=lambda: {severity: 0 for severity in Severity},
        init=False,
        repr=False,
    )
    _limit_notice_added: bool = field(
        default=False,
        init=False,
        repr=False,
    )
    _reported_issue_bytes: int = field(
        default=0,
        init=False,
        repr=False,
    )

    @property
    def errors(self) -> list[IntakeIssue]:
        return [issue for issue in self.issues if issue.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[IntakeIssue]:
        return [issue for issue in self.issues if issue.severity is Severity.WARNING]

    @property
    def infos(self) -> list[IntakeIssue]:
        return [issue for issue in self.issues if issue.severity is Severity.INFO]

    @property
    def total_error_count(self) -> int:
        return len(self.errors) + self._suppressed_issue_counts[Severity.ERROR]

    @property
    def total_warning_count(self) -> int:
        return len(self.warnings) + self._suppressed_issue_counts[Severity.WARNING]

    @property
    def total_info_count(self) -> int:
        return len(self.infos) + self._suppressed_issue_counts[Severity.INFO]

    @property
    def status(self) -> IntakeStatus:
        if self.total_error_count:
            return IntakeStatus.REJECTED
        if self.total_warning_count:
            return IntakeStatus.ACCEPTED_WITH_WARNINGS
        return IntakeStatus.ACCEPTED

    def add(
        self,
        code: str,
        severity: Severity,
        message: str,
        *,
        file: str | Path | None = None,
        json_path: str | None = None,
        expected: Any = None,
        actual: Any = None,
        hint: str | None = None,
    ) -> None:
        if hint is None:
            hint = (
                "Correct the reported value or reference, then rerun intake."
                if severity is not Severity.INFO
                else "No action is required unless this information is unexpected."
            )
        issue = IntakeIssue(
            code=_bounded_text(code, 256),
            severity=severity,
            message=_bounded_text(message),
            file=(
                _bounded_text(str(file))
                if file is not None
                else None
            ),
            json_path=(
                _bounded_text(json_path)
                if json_path is not None
                else None
            ),
            expected=_bounded_evidence(expected),
            actual=_bounded_evidence(actual),
            hint=_bounded_text(hint),
        )
        issue_bytes = len(
            json.dumps(
                issue.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        if (
            len(self.issues) >= MAX_REPORTED_ISSUES
            or self._reported_issue_bytes + issue_bytes
            > MAX_REPORTED_ISSUE_BYTES
        ):
            self._suppressed_issue_counts[severity] += 1
            if not self._limit_notice_added:
                self._limit_notice_added = True
                notice = IntakeIssue(
                    code="REPORT_ISSUE_LIMIT_EXCEEDED",
                    severity=Severity.ERROR,
                    message=(
                        "The intake report reached its bounded finding count "
                        "or byte budget; additional findings were suppressed."
                    ),
                    expected={
                        "maximum_reported_findings": MAX_REPORTED_ISSUES,
                        "maximum_reported_issue_bytes": (
                            MAX_REPORTED_ISSUE_BYTES
                        ),
                    },
                    actual={
                        "reported_findings_before_notice": len(self.issues),
                        "reported_issue_bytes_before_notice": (
                            self._reported_issue_bytes
                        ),
                    },
                    hint=(
                        "Fix the reported findings first, regenerate the package, "
                        "and rerun intake to reveal any remaining issues."
                    ),
                )
                self.issues.append(notice)
                self._reported_issue_bytes += len(
                    json.dumps(
                        notice.to_dict(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                )
            return
        self.issues.append(issue)
        self._reported_issue_bytes += issue_bytes

    def add_check(
        self,
        name: str,
        status: str,
        *,
        errors_added: int = 0,
        warnings_added: int = 0,
        note: str | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "name": name,
            "status": status,
            "errors_added": errors_added,
            "warnings_added": warnings_added,
        }
        if note:
            record["note"] = note
        self.checks.append(record)

    def to_dict(self) -> dict[str, Any]:
        code_counts: dict[str, int] = {}
        for issue in self.issues:
            code_counts[issue.code] = code_counts.get(issue.code, 0) + 1
        return {
            "schema_version": "0.1",
            "status": self.status.value,
            "package_dir": self.package_dir,
            "requirements_file": self.requirements_file,
            "generated_at": self.generated_at,
            "summary": {
                "error_count": self.total_error_count,
                "warning_count": self.total_warning_count,
                "info_count": self.total_info_count,
                "reported_issue_count": len(self.issues),
                "reported_issue_bytes": self._reported_issue_bytes,
                "reported_issue_byte_limit": MAX_REPORTED_ISSUE_BYTES,
                "suppressed_issue_count": sum(
                    self._suppressed_issue_counts.values()
                ),
                "suppressed_issue_counts": {
                    severity.value: self._suppressed_issue_counts[severity]
                    for severity in Severity
                    if self._suppressed_issue_counts[severity]
                },
                "issue_code_counts": dict(sorted(code_counts.items())),
            },
            "capabilities": self.capabilities,
            "package_summary": self.package_summary,
            "checks": self.checks,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )

    def write_json(self, path: str | Path) -> None:
        output = Path(path)
        payload = (self.to_json() + "\n").encode("utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(
            f".{output.name}.{uuid.uuid4().hex}.tmp"
        )
        descriptor: int | None = None
        write_error: OSError | None = None
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_BINARY", 0)
            descriptor = os.open(temporary, flags, 0o600)
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if not isinstance(written, int) or written <= 0:
                    raise OSError("report write made no forward progress")
                offset += written
            os.fsync(descriptor)
        except OSError as exc:
            write_error = exc
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError as exc:
                    write_error = write_error or exc

        try:
            if write_error is not None:
                raise write_error
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)

    def exit_code(self, *, fail_on_warning: bool = False) -> int:
        if self.status is IntakeStatus.REJECTED:
            return 2
        if fail_on_warning and self.status is IntakeStatus.ACCEPTED_WITH_WARNINGS:
            return 1
        return 0

    def render_text(self) -> str:
        lines = [
            f"Design package intake: {self.status.value}",
            f"Package: {self.package_dir}",
            f"Requirements: {self.requirements_file or 'NOT PROVIDED'}",
            (
                f"Issues: {self.total_error_count} error(s), "
                f"{self.total_warning_count} warning(s), "
                f"{self.total_info_count} info"
            ),
        ]
        suppressed_total = sum(self._suppressed_issue_counts.values())
        if suppressed_total:
            lines.append(
                "Report bounds: "
                f"{len(self.issues)} finding(s) shown, "
                f"{suppressed_total} suppressed; fix and rerun for the rest."
            )
        if self.capabilities:
            lines.append("Capabilities:")
            for key, value in self.capabilities.items():
                lines.append(f"  - {key}: {value}")
        if self.issues:
            lines.append("Findings:")
        for index, issue in enumerate(self.issues, start=1):
            location = ""
            if issue.file:
                location = f" [{issue.file}"
                if issue.json_path:
                    location += f":{issue.json_path}"
                location += "]"
            lines.append(
                f"{index}. {issue.severity.value} {issue.code}{location}: "
                f"{issue.message}"
            )
            if issue.expected is not None:
                lines.append(f"   Expected: {issue.expected}")
            if issue.actual is not None:
                lines.append(f"   Actual: {issue.actual}")
            if issue.hint:
                lines.append(f"   Fix: {issue.hint}")
        return "\n".join(lines)
