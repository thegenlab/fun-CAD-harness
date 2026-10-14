"""Run the FunCAD-Harness unit tests with the CAD environment's Python.

    python scripts/run_tests.py [-v]

Tests listed in simulation_ready_cad_workflow/validation_pipeline/tests/known_failures.txt
already fail upstream; they are reported as expected failures, and one that passes
is listed after the run without failing it.
"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "simulation_ready_cad_workflow/validation_pipeline/tests"
PREFIX = "simulation_ready_cad_workflow.validation_pipeline.tests."


def known_failures():
    lines = (TESTS / "known_failures.txt").read_text(encoding="utf-8").splitlines()
    return {line.split("#", 1)[0].strip() for line in lines} - {""}


def mark(suite, known):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            mark(test, known)
        elif test.id().removeprefix(PREFIX) in known:
            method = getattr(test, test._testMethodName)
            def expected(*args, _method=method, **kwargs):
                return _method(*args, **kwargs)
            expected.__unittest_expecting_failure__ = True
            setattr(test, test._testMethodName, expected)


def main():
    sys.path.insert(0, str(ROOT))
    suite = unittest.defaultTestLoader.discover(str(TESTS), top_level_dir=str(ROOT))
    mark(suite, known_failures())
    result = unittest.TextTestRunner(verbosity=2 if "-v" in sys.argv else 1).run(suite)
    # Some known failures depend on the environment; one that passes here is reported, not failed.
    for test in result.unexpectedSuccesses:
        print("Known failure passes here; remove it from known_failures.txt if it passes everywhere:", test.id())
    return 1 if result.failures or result.errors else 0


if __name__ == "__main__":
    sys.exit(main())
