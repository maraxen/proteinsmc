"""Data structures for annealing schedules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypedDict, Unpack


class AnnealingKwargs(TypedDict):
  """TypedDict for the parameters of an annealing schedule function."""

  current_step: int
  _context: Any | None


AnnealingFuncSignature = Callable[[Unpack[AnnealingKwargs]], float]


@dataclass(frozen=True)
class AnnealingConfig:
  """Configuration for an annealing schedule.

  Attributes:
      schedule_fn: String that defines the annealing schedule.
      beta_max: Maximum value for beta.
      n_steps: Number of steps over which to anneal.
      schedule_args: Additional arguments for the schedule function.

  """

  annealing_fn: str
  beta_min: float = 0.1
  beta_max: float = 1.0
  n_steps: int = 1000
  kwargs: dict[str, Any] = field(default_factory=dict)
