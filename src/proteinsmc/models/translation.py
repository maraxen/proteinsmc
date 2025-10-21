"""Type definitions for translation functions."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypedDict, Unpack

from proteinsmc.models.types import EvoSequence

if TYPE_CHECKING:
  from jaxtyping import Array, PRNGKeyArray


class TranslateKwargs(TypedDict):
  """TypedDict for the parameters of a translation function."""

  sequence: EvoSequence
  _key: PRNGKeyArray | None
  _context: Array | None


TranslateFuncSignature = Callable[[Unpack[TranslateKwargs]], EvoSequence]
