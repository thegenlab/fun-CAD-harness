"""Static architecture tests locking the Harness / Loop / contracts boundary.

These tests parse every production module with ``ast`` and fail on any
import that crosses a forbidden boundary, including lazy function-level
imports.  A separate text-level check proves the Harness package does not
smuggle a Loop dependency through string-based imports.

Dependency rules (direction of allowed imports):

    react_loop  ->  harness (package root only), contracts, mechanism,
                    intake, runtime, planning_templates, run_layout, ...
    harness     ->  contracts, mechanism, intake, runtime, dashboard? NO:
                    harness may import dashboard.visuals lazily? NO — the
                    visualization publisher is Harness-owned output today;
                    see ALLOWED_EXCEPTIONS below for the single sanctioned
                    edge until the visuals module moves.
    contracts   ->  mechanism.io (hash/json utilities), intake,
                    planning_templates, run_layout only
    mechanism / runtime / intake / planning_templates
                ->  never react_loop, never harness, never dashboard
    dashboard   ->  contracts and read-only helpers; the unified local human
                    board has an exact allowlist of Loop projection and event
                    ingress adapters; never harness
"""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PREFIX = "simulation_ready_cad_workflow.validation_pipeline"

# Top-level component a module belongs to, by first path part below
# validation_pipeline.
FORBIDDEN_TARGETS: dict[str, frozenset[str]] = {
    "harness": frozenset({"react_loop"}),
    "contracts": frozenset({"react_loop", "harness", "runtime", "dashboard"}),
    "mechanism": frozenset({"react_loop", "harness", "dashboard"}),
    "runtime": frozenset({"react_loop", "harness", "dashboard"}),
    "intake": frozenset({"react_loop", "harness", "dashboard"}),
    "planning_templates": frozenset({"react_loop", "harness", "dashboard"}),
    "dashboard": frozenset({"react_loop", "harness"}),
    # A benchmark bundle must run without the agentic Loop: the exporter and
    # the clean-directory runner may consume contracts, intake, mechanism,
    # and the public harness surface, but never react_loop or dashboard.
    "benchmark": frozenset({"react_loop", "dashboard"}),
}

# The Harness publishes dashboard visual summaries as part of its result
# today (``_publish_full_visualization``).  That one edge is sanctioned
# until visual generation moves behind an observability contract; nothing
# else in the Harness may import dashboard code.
ALLOWED_EXCEPTIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("harness.engine", "dashboard.visuals"),
        # Failure-evidence videos reuse the dashboard's trace renderer.
        # Same category as above: a visual-generation edge sanctioned
        # until rendering moves behind an observability contract.
        ("mechanism.failure_evidence", "dashboard.trace_pair_video"),
        # The unified Loop / Loop Loop board is the local human surface for
        # safe role transcripts, resource decisions, and maintenance
        # proposals.  Keep each Loop edge explicit so a new dashboard import
        # cannot silently turn into a broad orchestration dependency.
        ("dashboard.maintenance_view", "react_loop.role_runtime"),
        ("dashboard.maintenance_routes", "react_loop.event_kernel"),
        ("dashboard.maintenance_routes", "react_loop.models"),
        ("dashboard.maintenance_routes", "react_loop.role_runtime"),
        ("dashboard.maintenance_routes", "react_loop.visible_session_gateway"),
        ("dashboard.session_view", "react_loop.models"),
        ("dashboard.session_view", "react_loop.role_runtime"),
    }
)

# react_loop may consume the Harness only through its public package
# root (the adapter factory and the visual-review evaluator re-export).
LOOP_ALLOWED_HARNESS_TARGETS = frozenset({"harness"})


def _module_name(path: Path) -> str:
    relative = path.relative_to(PIPELINE_ROOT)
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _iter_production_modules() -> list[Path]:
    modules: list[Path] = []
    for path in sorted(PIPELINE_ROOT.rglob("*.py")):
        relative = path.relative_to(PIPELINE_ROOT)
        first = relative.parts[0]
        if first in {"tests", "__pycache__"} or "__pycache__" in relative.parts:
            continue
        modules.append(path)
    return modules


def _resolve_relative(module: str, node: ast.ImportFrom) -> str | None:
    """Resolve a relative import to a dotted path below validation_pipeline."""

    if node.level == 0:
        target = node.module or ""
        prefix = f"{PACKAGE_PREFIX}."
        if target.startswith(prefix):
            return target[len(prefix):]
        return None
    parts = module.split(".")
    # A module ``a.b.c`` with level=1 imports from package ``a.b``.
    anchor = parts[: len(parts) - node.level] if not _is_package(module) else parts[: len(parts) - node.level + 1]
    base = anchor
    if node.module:
        base = anchor + node.module.split(".")
    if not base:
        return None
    return ".".join(base)


def _is_package(module: str) -> bool:
    return (PIPELINE_ROOT / Path(*module.split("."))).is_dir()


def _imported_targets(path: Path) -> list[str]:
    module = _module_name(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                prefix = f"{PACKAGE_PREFIX}."
                if alias.name.startswith(prefix):
                    targets.append(alias.name[len(prefix):])
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_relative(module, node)
            if resolved is not None:
                targets.append(resolved)
    return targets


class ArchitectureBoundaryTest(unittest.TestCase):
    def test_no_forbidden_cross_component_imports(self) -> None:
        violations: list[str] = []
        for path in _iter_production_modules():
            module = _module_name(path)
            component = module.split(".")[0] if module else ""
            forbidden = FORBIDDEN_TARGETS.get(component)
            if forbidden is None:
                continue
            for target in _imported_targets(path):
                target_component = target.split(".")[0]
                if target_component not in forbidden:
                    continue
                if (module, target) in ALLOWED_EXCEPTIONS:
                    continue
                violations.append(f"{module} -> {target}")
        self.assertEqual(
            violations,
            [],
            "forbidden cross-component imports:\n" + "\n".join(violations),
        )

    def test_loop_uses_only_public_harness_surface(self) -> None:
        violations: list[str] = []
        for path in _iter_production_modules():
            module = _module_name(path)
            if not module.startswith("react_loop"):
                continue
            for target in _imported_targets(path):
                if (
                    target.split(".")[0] == "harness"
                    and target not in LOOP_ALLOWED_HARNESS_TARGETS
                ):
                    violations.append(f"{module} -> {target}")
        self.assertEqual(
            violations,
            [],
            "the Loop must consume the Harness only through its public "
            "package root:\n" + "\n".join(violations),
        )

    def test_harness_sources_never_mention_react_loop(self) -> None:
        """No hidden coupling via strings, importlib, or comments."""

        offenders: list[str] = []
        for path in sorted((PIPELINE_ROOT / "harness").rglob("*.py")):
            if "react_loop" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(PIPELINE_ROOT)))
        self.assertEqual(
            offenders,
            [],
            "harness sources must not reference react_loop at all: "
            + ", ".join(offenders),
        )

    def test_benchmark_sources_never_mention_react_loop(self) -> None:
        """Bundles must run without the Loop; no hidden string imports."""

        offenders: list[str] = []
        for path in sorted((PIPELINE_ROOT / "benchmark").rglob("*.py")):
            if "react_loop" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(PIPELINE_ROOT)))
        self.assertEqual(
            offenders,
            [],
            "benchmark sources must not reference react_loop at all: "
            + ", ".join(offenders),
        )

    def test_contracts_never_mention_harness_package(self) -> None:
        """Contracts define the boundary; they must not know the engine."""

        offenders: list[str] = []
        for path in sorted((PIPELINE_ROOT / "contracts").rglob("*.py")):
            for target in _imported_targets(path):
                if target.split(".")[0] == "harness":
                    offenders.append(str(path.relative_to(PIPELINE_ROOT)))
        self.assertEqual(offenders, [])

    def test_production_components_never_import_stage_replay(self) -> None:
        violations: list[str] = []
        for path in _iter_production_modules():
            module = _module_name(path)
            component = module.split(".")[0] if module else ""
            if component in {"replay", "tools"}:
                continue
            for target in _imported_targets(path):
                if (
                    target == "replay"
                    or target.startswith("replay.")
                    or target == "tools.stage_replay"
                    or target.startswith("tools.stage_replay.")
                ):
                    violations.append(f"{module} -> {target}")
        self.assertEqual(
            violations,
            [],
            "production components must not depend on diagnostic replay:\n"
            + "\n".join(violations),
        )

    def test_replay_has_only_the_pure_error_routing_loop_edge(self) -> None:
        violations: list[str] = []
        for path in sorted((PIPELINE_ROOT / "replay").rglob("*.py")):
            module = _module_name(path)
            for target in _imported_targets(path):
                if target == "react_loop.error_routing":
                    continue
                if target.split(".")[0] in {"react_loop", "harness"}:
                    violations.append(f"{module} -> {target}")
        self.assertEqual(
            violations,
            [],
            "replay may import only react_loop.error_routing from the Loop "
            "and must never import the Harness:\n"
            + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
