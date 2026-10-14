from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys
import types
from typing import Any


def load_pychrono_core() -> Any:
    """Load headless Chrono core without importing optional GUI modules."""

    loaded = sys.modules.get("pychrono.core")
    if loaded is not None:
        return loaded

    discovered = importlib.util.find_spec("pychrono")
    if discovered is None or not discovered.submodule_search_locations:
        raise ModuleNotFoundError("pychrono package is not installed")
    locations = list(discovered.submodule_search_locations)
    package = types.ModuleType("pychrono")
    package.__file__ = discovered.origin
    package.__package__ = "pychrono"
    package.__path__ = locations
    package.__spec__ = importlib.machinery.ModuleSpec(
        "pychrono", loader=None, is_package=True
    )
    package.__spec__.submodule_search_locations = locations
    sys.modules["pychrono"] = package

    core = importlib.import_module("pychrono.core")
    data = (Path(locations[0]) / "../../../Library/data").resolve()
    if hasattr(core, "SetChronoDataPath"):
        core.SetChronoDataPath(str(data) + os.sep)
    return core
