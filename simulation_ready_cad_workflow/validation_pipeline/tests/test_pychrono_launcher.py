from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simulation_ready_cad_workflow.validation_pipeline.mechanism.launcher import (
    DEFAULT_PYCHRONO_ENVIRONMENT,
    PYCHRONO_ENVIRONMENT_ENV,
    PYCHRONO_EXECUTABLE_ENV,
    PYCHRONO_LAUNCHER_ENV,
    PyChronoLauncher,
    PyChronoLauncherKind,
    active_pychrono_launcher,
    resolve_pychrono_launcher,
    use_pychrono_launcher,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    read_json,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismValidationError,
)


class PyChronoLauncherTests(unittest.TestCase):
    def test_windows_default_preserves_historical_conda_command(self) -> None:
        launcher = resolve_pychrono_launcher(
            environ={},
            platform_name="nt",
            python_executable=r"C:\Python312\python.exe",
        )

        self.assertEqual(launcher.kind, PyChronoLauncherKind.CONDA)
        self.assertEqual(
            launcher.python_command("-m", "example.worker"),
            (
                "conda.exe",
                "run",
                "-n",
                DEFAULT_PYCHRONO_ENVIRONMENT,
                "python",
                "-m",
                "example.worker",
            ),
        )

    def test_posix_requires_explicit_launcher(self) -> None:
        with self.assertRaises(MechanismValidationError) as caught:
            resolve_pychrono_launcher(
                environ={},
                platform_name="posix",
                python_executable="/opt/harness/bin/python",
            )

        self.assertEqual(
            caught.exception.code,
            "PYCHRONO_LAUNCHER_REQUIRED",
        )

    def test_explicit_python_uses_selected_interpreter(self) -> None:
        launcher = resolve_pychrono_launcher(
            "python",
            executable="/opt/physics/bin/python",
            environ={},
            platform_name="posix",
        )
        self.assertEqual(launcher.kind, PyChronoLauncherKind.PYTHON)
        self.assertEqual(
            launcher.python_command("-m", "example.worker"),
            ("/opt/physics/bin/python", "-m", "example.worker"),
        )

    def test_micromamba_supports_name_and_absolute_prefix(self) -> None:
        named = resolve_pychrono_launcher(
            "micromamba",
            executable="/usr/local/bin/micromamba",
            environment="sr-chrono-py312",
            environ={},
            platform_name="posix",
        )
        prefixed = resolve_pychrono_launcher(
            "micromamba",
            executable="micromamba",
            environment="/opt/envs/sr-chrono-py312",
            environ={},
            platform_name="posix",
        )

        self.assertEqual(
            named.python_command("-V")[:5],
            (
                "/usr/local/bin/micromamba",
                "run",
                "-n",
                "sr-chrono-py312",
                "python",
            ),
        )
        self.assertEqual(
            prefixed.python_command("-V")[:5],
            (
                "micromamba",
                "run",
                "-p",
                "/opt/envs/sr-chrono-py312",
                "python",
            ),
        )

    def test_operator_environment_variables_are_honored(self) -> None:
        launcher = resolve_pychrono_launcher(
            environ={
                PYCHRONO_LAUNCHER_ENV: "micromamba",
                PYCHRONO_EXECUTABLE_ENV: "/opt/bin/micromamba",
                PYCHRONO_ENVIRONMENT_ENV: "sr-chrono-py312",
            },
            platform_name="posix",
        )

        self.assertEqual(
            launcher.to_dict(),
            {
                "kind": "micromamba",
                "executable": "/opt/bin/micromamba",
                "environment": "sr-chrono-py312",
            },
        )

    def test_invalid_ambiguous_or_shell_configuration_fails_closed(self) -> None:
        invalid = (
            lambda: resolve_pychrono_launcher(
                "python",
                environment="sr-chrono-py312",
                environ={},
            ),
            lambda: PyChronoLauncher(
                PyChronoLauncherKind.MICROMAMBA,
                "bash",
                "sr-chrono-py312",
            ),
            lambda: PyChronoLauncher(
                PyChronoLauncherKind.CONDA,
                "conda",
                "relative/env",
            ),
        )

        for build in invalid:
            with self.subTest(build=build), self.assertRaises(
                MechanismValidationError
            ) as caught:
                build()
            self.assertEqual(
                caught.exception.code,
                "PYCHRONO_LAUNCHER_INVALID",
            )

    def test_launcher_context_restores_the_previous_binding(self) -> None:
        outer = PyChronoLauncher(
            PyChronoLauncherKind.PYTHON,
            "/opt/outer/bin/python",
        )
        inner = PyChronoLauncher(
            PyChronoLauncherKind.MICROMAMBA,
            "micromamba",
            "sr-chrono-py312",
        )

        with use_pychrono_launcher(outer):
            self.assertEqual(active_pychrono_launcher(), outer)
            with use_pychrono_launcher(inner):
                self.assertEqual(active_pychrono_launcher(), inner)
            self.assertEqual(active_pychrono_launcher(), outer)


@unittest.skip("removed pre-ReAct mechanism CLI")
class PyChronoLauncherCliTests(unittest.TestCase):
    def test_launcher_options_parse_for_pipeline_commands(self) -> None:
        args = build_parser().parse_args(
            [
                "plan",
                "--package",
                "package",
                "--intake-report",
                "intake.json",
                "--workspace-root",
                ".",
                "--output-dir",
                "run",
                "--pychrono-launcher",
                "micromamba",
                "--pychrono-executable",
                "/opt/bin/micromamba",
                "--pychrono-environment",
                "sr-chrono-py312",
            ]
        )

        self.assertEqual(args.pychrono_launcher, "micromamba")
        self.assertEqual(args.pychrono_executable, "/opt/bin/micromamba")
        self.assertEqual(args.pychrono_environment, "sr-chrono-py312")

    def test_run_forwards_one_resolved_launcher_to_orchestrator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "cli.run_pipeline",
                return_value={"status": "TESTED"},
            ) as run_pipeline, contextlib.redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "run",
                        "--package",
                        str(root / "package"),
                        "--intake-report",
                        str(root / "intake.json"),
                        "--workspace-root",
                        str(root),
                        "--output-dir",
                        str(root / "run"),
                        "--pychrono-launcher",
                        "micromamba",
                        "--pychrono-executable",
                        "/opt/bin/micromamba",
                        "--pychrono-environment",
                        "sr-chrono-py312",
                    ]
                )

        self.assertEqual(exit_code, 0)
        launcher = run_pipeline.call_args.kwargs["pychrono_launcher"]
        self.assertEqual(launcher.kind, PyChronoLauncherKind.MICROMAMBA)
        self.assertEqual(launcher.executable, "/opt/bin/micromamba")
        self.assertEqual(launcher.environment, "sr-chrono-py312")

    def test_finalize_binds_launcher_for_whole_machine_dispatch(self) -> None:
        observed: list[PyChronoLauncher] = []
        finalize_calls: list[dict[str, object]] = []

        def finalize(**kwargs: object) -> dict[str, object]:
            observed.append(active_pychrono_launcher())
            finalize_calls.append(kwargs)
            return {
                "schema_version": "0.1",
                "status": "NOT_CERTIFIED",
                "mechanisms": [],
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run").mkdir()
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "cli.finalize_dispatch",
                side_effect=finalize,
            ), contextlib.redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "finalize",
                        "--workspace-root",
                        str(root),
                        "--output-dir",
                        str(root / "run"),
                        "--pychrono-launcher",
                        "python",
                        "--pychrono-executable",
                        "/opt/physics/bin/python",
                    ]
                )
            recorded = read_json(root / "run" / "run_status.json")

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(observed), 1)
        self.assertIs(finalize_calls[0]["commit_terminal"], False)
        self.assertEqual(recorded["status"], "NOT_CERTIFIED")
        self.assertIn("artifact_inventory_sha256", recorded)
        self.assertEqual(
            observed[0].python_command("-V"),
            ("/opt/physics/bin/python", "-V"),
        )

    def test_early_failure_is_attempt_only_relative_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = main(
                    [
                        "plan",
                        "--package",
                        str(root / "package"),
                        "--intake-report",
                        str(root / "intake.json"),
                        "--workspace-root",
                        str(root),
                        "--output-dir",
                        str(root / "run"),
                        "--pychrono-launcher",
                        "python",
                        "--pychrono-environment",
                        "invalid-for-python",
                    ]
                )
            failure = read_json(root / "run" / "failure.json")

        self.assertEqual(exit_code, 2)
        self.assertEqual(failure["stage"], "plan")
        self.assertEqual(
            failure["error_code"],
            "PYCHRONO_LAUNCHER_INVALID",
        )
        self.assertEqual(failure["failure_artifact"], "run/failure.json")
        self.assertEqual(failure["outcome_scope"], "iteration_attempt_only")
        self.assertFalse(failure["terminal_authority"])
        self.assertTrue(failure["terminal_publication_skipped"])
        self.assertFalse((root / "run" / "run_status.json").exists())
        self.assertFalse((root / "run" / "artifacts.sha256.json").exists())
        self.assertIn("PYCHRONO_LAUNCHER_INVALID", stderr.getvalue())

    def test_failure_commit_error_does_not_mask_original_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stderr = io.StringIO()
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "cli.commit_terminal_run_status",
                side_effect=OSError("synthetic terminal commit failure"),
            ), contextlib.redirect_stderr(stderr):
                exit_code = main(
                    [
                        "plan",
                        "--package",
                        str(root / "package"),
                        "--intake-report",
                        str(root / "intake.json"),
                        "--workspace-root",
                        str(root),
                        "--output-dir",
                        str(root / "run"),
                        "--pychrono-launcher",
                        "python",
                        "--pychrono-environment",
                        "invalid-for-python",
                    ]
                )

        self.assertEqual(exit_code, 2)
        self.assertIn("PYCHRONO_LAUNCHER_INVALID", stderr.getvalue())
        self.assertNotIn(
            "synthetic terminal commit failure",
            stderr.getvalue(),
        )

    def test_finalize_with_dispatch_ready_stays_running(self) -> None:
        provisional = {
            "schema_version": "0.1",
            "status": "KINEMATIC_CERTIFIED",
            "last_completed_stage": "verdict_and_certificate",
            "mechanisms": [
                {
                    "capsule_id": "fixture",
                    "whole_machine_dispatch": {
                        "queue_manifest": (
                            "run/results/fixture/whole_machine/"
                            "queue_manifest.json"
                        )
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "cli.finalize_dispatch",
                return_value=provisional,
            ) as finalize, contextlib.redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "finalize",
                        "--workspace-root",
                        str(root),
                        "--output-dir",
                        str(root / "run"),
                        "--pychrono-launcher",
                        "python",
                        "--pychrono-executable",
                        "/opt/physics/bin/python",
                    ]
                )
            recorded = read_json(root / "run" / "run_status.json")
            index_exists = (root / "run" / "artifacts.sha256.json").exists()

        self.assertEqual(exit_code, 0)
        finalize.assert_called_once_with(
            workspace_root=root,
            output_dir=root / "run",
            commit_terminal=False,
        )
        self.assertEqual(recorded["status"], "RUNNING")
        self.assertEqual(
            recorded["provisional_verdict"],
            "KINEMATIC_CERTIFIED",
        )
        self.assertTrue(recorded["whole_machine_dispatch_ready"])
        self.assertFalse(index_exists)


if __name__ == "__main__":
    unittest.main()
