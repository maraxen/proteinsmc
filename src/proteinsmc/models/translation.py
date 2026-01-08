"""Type definitions for translation functions."""

from __future__ import annotations

from collections.abc import Callable

from jaxtyping import Array, Bool, PRNGKeyArray, PyTree

from proteinsmc.models.types import EvoSequence

TranslateFuncSignature = Callable[
  [EvoSequence, PRNGKeyArray | None, PyTree | Array | None], tuple[EvoSequence, Bool]
]
