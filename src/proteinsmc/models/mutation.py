"""Mutation related types."""

from __future__ import annotations

from typing import Callable

from jaxtyping import PRNGKeyArray

from proteinsmc.models.types import EvoSequence

MutationFn = Callable[[PRNGKeyArray, EvoSequence], EvoSequence]
