"""Validation stages for simulation-ready CAD design packages."""

# Pip OCP wheels may bundle an older libexpat under the same SONAME.  Load the
# Python-runtime-matched extension before any OCCT import can claim that name.
import xml.parsers.expat  # noqa: F401

__all__ = ["validate_design_package"]


def __getattr__(name):
    # Peripheral archive tools must remain usable without the CAD dependencies.
    if name == 'validate_design_package':
        from .intake import validate_design_package
        return validate_design_package
    raise AttributeError(name)
