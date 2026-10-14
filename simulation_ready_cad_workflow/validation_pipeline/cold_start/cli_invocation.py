"""Resolve installed PowerShell CLI shims without a command-string shell."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys


def cli_command(command: list[str]) -> list[str]:
    if sys.platform != "win32":
        return command
    executable = command[0]
    resolved = shutil.which(executable)
    if resolved and Path(resolved).suffix.lower() in {".cmd", ".bat"} and not Path(executable).suffix:
        # Installers can provide both shell shims. PowerShell selects .ps1;
        # shutil.which selects .CMD via PATHEXT, which is not a native binary.
        sibling = Path(resolved).with_suffix(".ps1")
        if sibling.is_file():
            resolved = str(sibling)
    if resolved is None:
        # PowerShell finds .ps1 scripts on PATH even when PATHEXT omits .PS1.
        script = executable if executable.lower().endswith(".ps1") else executable + ".ps1"
        if Path(executable).parent != Path(".") or Path(executable).is_absolute():
            candidates = [Path(script)]
        else:
            candidates = [Path(directory.strip('"')) / script for directory in os.get_exec_path()]
        resolved = next((str(path.resolve()) for path in candidates if path.is_file()), None)
    if resolved is None or Path(resolved).suffix.lower() != ".ps1":
        return command
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        raise FileNotFoundError("PowerShell is required to launch the installed CLI script: " + resolved)
    return [shell, "-NoProfile", "-File", resolved, *command[1:]]
