from __future__ import annotations


MAX_WINDOWS_ENVIRONMENT_UTF16_UNITS = 32_767


def windows_environment_utf16_units(
    environment: dict[str, str],
) -> int:
    """Return CreateProcess environment-block size in UTF-16 code units."""
    units = 1
    for name, value in environment.items():
        entry = f"{name}={value}" + chr(0)
        units += len(entry.encode("utf-16-le")) // 2
    return units


def update_environment_case_insensitively(
    environment: dict[str, str],
    updates: dict[str, str],
) -> None:
    """Apply controlled variables without Windows case-colliding aliases."""
    replaced_names = {name.casefold() for name in updates}
    for existing_name in tuple(environment):
        if existing_name.casefold() in replaced_names:
            del environment[existing_name]
    environment.update(updates)
