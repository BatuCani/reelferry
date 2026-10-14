"""Reelferry: an agent-first clip studio that runs on your own machine."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("reelferry")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0"

__all__ = ["__version__"]
