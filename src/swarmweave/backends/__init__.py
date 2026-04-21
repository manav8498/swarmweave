"""Backend implementations for the shared-context layer."""

from swarmweave.backends.base import Backend
from swarmweave.backends.local import LocalBackend

__all__ = ["Backend", "LocalBackend"]
