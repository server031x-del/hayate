"""HAYATE consumer GPU runtime foundation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hayate-runtime")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]

