from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from simulation_ready_cad_workflow.validation_pipeline.runtime.machine_launch import (
    CAD_ENV_NAME,
    CHRONO_ENV_NAME,
    MACHINE_LAUNCH_SCHEMA,
    MachineLaunchError,
    ensure_machine_launch,
    kernel_spawn_environment,
    load_machine_launch,
    pychrono_launcher_from_recipe,
    recipe_path,
)


def _write_env(root: Path, name: str) -> Path:
    prefix = root / "envs" / name
    (prefix / "bin").mkdir(parents=True)
    (prefix / "lib").mkdir()
    python = prefix / "bin" / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    return prefix


def _healthy(*_args, **_kwargs) -> None:
    return None


class MachineLaunchTests(unittest.TestCase):
    def test_first_run_writes_workspace_recipe_from_existing_envs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "repo"
            workspace.mkdir()
            mamba = Path(temporary) / "mamba"
            _write_env(mamba, CAD_ENV_NAME)
            _write_env(mamba, CHRONO_ENV_NAME)
            installs: list[tuple[bool, bool]] = []

            def installer(**kwargs):
                installs.append((kwargs["cad_exists"], kwargs["chrono_exists"]))

            recipe = ensure_machine_launch(
                workspace_root=workspace,
                mamba_root=mamba,
                probe_imports=_healthy,
                install_lock=installer,
            )

            self.assertEqual(recipe["schema_version"], MACHINE_LAUNCH_SCHEMA)
            self.assertEqual(
                recipe["cad_python"],
                str(mamba / "envs" / CAD_ENV_NAME / "bin" / "python"),
            )
            self.assertEqual(
                recipe["pychrono_executable"],
                str(mamba / "envs" / CHRONO_ENV_NAME / "bin" / "python"),
            )
            self.assertEqual(recipe["pychrono_launcher"], "python")
            self.assertEqual(installs, [])
            loaded = load_machine_launch(workspace)
            self.assertEqual(loaded["cad_python"], recipe["cad_python"])
            launcher = pychrono_launcher_from_recipe(recipe)
            self.assertEqual(launcher.executable, recipe["pychrono_executable"])

    def test_second_run_reuses_healthy_recipe_without_install(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "repo"
            workspace.mkdir()
            mamba = Path(temporary) / "mamba"
            _write_env(mamba, CAD_ENV_NAME)
            _write_env(mamba, CHRONO_ENV_NAME)
            first = ensure_machine_launch(
                workspace_root=workspace,
                mamba_root=mamba,
                probe_imports=_healthy,
                install_lock=lambda **_kwargs: None,
            )
            probes = []

            def probe(python, *, modules, ld_library_path):
                probes.append((str(python), modules, ld_library_path))
                return None

            second = ensure_machine_launch(
                workspace_root=workspace,
                mamba_root=Path(temporary) / "other",
                probe_imports=probe,
                install_lock=lambda **_kwargs: (_ for _ in ()).throw(
                    AssertionError("must not reinstall a healthy recipe")
                ),
            )
            self.assertEqual(second["cad_python"], first["cad_python"])
            self.assertEqual(len(probes), 2)

    def test_missing_envs_create_from_lock_then_write_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "repo"
            workspace.mkdir()
            mamba = Path(temporary) / "mamba"

            def installer(**kwargs):
                _write_env(kwargs["mamba_root"], CAD_ENV_NAME)
                _write_env(kwargs["mamba_root"], CHRONO_ENV_NAME)

            recipe = ensure_machine_launch(
                workspace_root=workspace,
                mamba_root=mamba,
                probe_imports=_healthy,
                install_lock=installer,
            )
            self.assertTrue(Path(recipe["cad_python"]).is_file())
            self.assertTrue(recipe_path(workspace).is_file())

    def test_missing_envs_fail_closed_when_install_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "repo"
            workspace.mkdir()
            with self.assertRaises(MachineLaunchError) as raised:
                ensure_machine_launch(
                    workspace_root=workspace,
                    mamba_root=Path(temporary) / "mamba",
                    install_if_missing=False,
                    probe_imports=_healthy,
                    install_lock=lambda **_kwargs: None,
                )
            self.assertEqual(raised.exception.code, "MACHINE_LAUNCH_ENV_MISSING")

    def test_unhealthy_recipe_updates_from_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "repo"
            workspace.mkdir()
            mamba = Path(temporary) / "mamba"
            _write_env(mamba, CAD_ENV_NAME)
            _write_env(mamba, CHRONO_ENV_NAME)
            ensure_machine_launch(
                workspace_root=workspace,
                mamba_root=mamba,
                probe_imports=_healthy,
                install_lock=lambda **_kwargs: None,
            )
            calls = []
            seen = {"n": 0}

            def probe(*_args, **_kwargs):
                seen["n"] += 1
                if seen["n"] <= 2:
                    return "missing psutil"
                return None

            def installer(**kwargs):
                calls.append(kwargs["cad_exists"])

            recipe = ensure_machine_launch(
                workspace_root=workspace,
                mamba_root=mamba,
                probe_imports=probe,
                install_lock=installer,
            )
            self.assertEqual(calls, [True])
            self.assertEqual(recipe["mamba_root"], str(mamba))

    def test_kernel_spawn_environment_prepends_cad_lib(self) -> None:
        env = kernel_spawn_environment(
            {"cad_library_directory": "/opt/cad/lib"},
            environ={"PATH": "/bin", "LD_LIBRARY_PATH": "/usr/lib"},
        )
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
        self.assertTrue(env["LD_LIBRARY_PATH"].startswith("/opt/cad/lib"))
        self.assertIn("/usr/lib", env["LD_LIBRARY_PATH"])


if __name__ == "__main__":
    unittest.main()
