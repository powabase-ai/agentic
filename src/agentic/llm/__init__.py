"""LiteLLM integration: setup, routing, accumulator, extractor."""

# Run global LiteLLM config FIRST on package import — before any other agentic
# code imports litellm or builds an Agent. The isort splits below preserve this
# ordering against `ruff check --fix`.
from . import setup  # noqa: F401 -- triggers setup.py global flag on cold import

# isort: split
import importlib as _importlib
import sys as _sys

from agentic.llm.streaming import (
    AbortedError,
    Message,
    StreamPartialError,
    StreamTruncationError,
    accumulate_stream,
)

# Drop `setup` from the namespace so `from agentic.llm import setup` falls
# through to __getattr__, which reloads setup.py and re-asserts the flag.
# This makes the import contract idempotent across arbitrary test ordering.
del setup  # type: ignore[name-defined]


def __getattr__(name):
    if name == "setup":
        mod = _sys.modules[f"{__name__}.setup"]
        _importlib.reload(mod)
        return mod
    raise AttributeError(f"module 'agentic.llm' has no attribute {name!r}")


__all__ = [
    "AbortedError",
    "Message",
    "StreamPartialError",
    "StreamTruncationError",
    "accumulate_stream",
]
