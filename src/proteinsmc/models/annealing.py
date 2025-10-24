"""Data structures for annealing schedules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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

  def __post_init__(self) -> None:
    """Post-initialization to validate types."""
    if not isinstance(self.annealing_fn, str):
      msg = f"Expected annealing_fn to be a string, got {type(self.annealing_fn)}"
      raise TypeError(msg)
    if not isinstance(self.beta_min, float):
      msg = f"Expected beta_min to be a float, got {type(self.beta_min)}"
      raise TypeError(msg)
    if not isinstance(self.beta_max, float):
      msg = f"Expected beta_max to be a float, got {type(self.beta_max)}"
      raise TypeError(msg)
    if not isinstance(self.n_steps, int):
      msg = f"Expected n_steps to be an integer, got {type(self.n_steps)}"
      raise TypeError(msg)
    if not isinstance(self.kwargs, dict):
      msg = f"Expected kwargs to be a dict, got {type(self.kwargs)}"
      raise TypeError(msg)
