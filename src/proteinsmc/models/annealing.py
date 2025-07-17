"""Data structures for SMC sampling algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (  # Added Any, Callable for clarity
  Any,
  Callable,
  ParamSpec,
  TypedDict,
  Unpack,
)

import jax
from jaxtyping import Array, Float, Int

from .registry_base import RegisteredFunction, Registry, RegistryItem

P = ParamSpec("P")
CurrentStepInt = Int[jax.Array, "current_step"]
ScheduleLenInt = Int[jax.Array, "schedule_len"]
MaxBetaFloat = Float[jax.Array, "max_beta"]
CurrentBetaFloat = Float[jax.Array, "current_beta"]


class AnnealingKwargs(TypedDict):
  """TypedDict for the parameters of an annealing function."""

  current_step: CurrentStepInt
  n_steps: ScheduleLenInt
  beta_max: MaxBetaFloat
  _context: Array | None


AnnealingFuncSignature = Callable[[Unpack[AnnealingKwargs]], CurrentBetaFloat]


@dataclass(frozen=True)
class RegisteredAnnealingFunction(RegisteredFunction):
  """Represents a registered annealing function."""


@dataclass(frozen=True)
class AnnealingRegistryItem(RegistryItem):
  """Represents an annealing function for beta values."""

  method_factory: Callable[
    ...,
    AnnealingFuncSignature,
  ]
  name: str = "unnamed_annealing_function"

  def __call__(
    self,
    *args: Any,  # noqa: ANN401
    **kwds: Any,  # noqa: ANN401
  ) -> AnnealingFuncSignature:
    """Call the annealing function factory to get the actual function."""
    return self.method_factory(*args, **kwds)


@dataclass
class AnnealingScheduleRegistry(Registry):
  """Registry for annealing schedules."""

  def __post_init__(self) -> None:
    """Initialize the registry."""
    super().__post_init__()
    if not all(isinstance(item, AnnealingRegistryItem) for item in self.items.values()):
      msg = "All items in AnnealingScheduleRegistry must be instances of AnnealingRegistryItem."
      raise TypeError(msg)


@dataclass(frozen=True)
class AnnealingScheduleConfig:
  """Configuration for an annealing schedule.

  Attributes:
      schedule_fn: String that defines the annealing schedule.
      beta_max: Maximum value for beta.
      n_steps: Number of steps over which to anneal.
      schedule_args: Additional arguments for the schedule function.

  """

  schedule_fn: str
  beta_max: float  # TODO: maybe use this directly?
  n_steps: int
  schedule_args: dict[str, Any] = field(default_factory=dict)

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility.

    All fields are treated as children as they can vary across instances.
    """
    children = (
      self.beta_max,
      self.n_steps,
    )
    aux_data = {"schedule_fn": self.schedule_fn, "schedule_args": self.schedule_args}
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, _aux_data: dict, children: tuple) -> AnnealingScheduleConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(
      beta_max=children[0],
      n_steps=children[1],
      schedule_fn=_aux_data["schedule_fn"],
      schedule_args=_aux_data["schedule_args"],
    )

  def __call__(
    self,
    registry: AnnealingScheduleRegistry,
  ) -> AnnealingFuncSignature:
    """Get the annealing function based on the configuration."""
    if self.schedule_fn not in registry:
      msg = f"Annealing schedule '{self.schedule_fn}' is not registered."
      raise ValueError(msg)

    return registry.get(self.schedule_fn)(
      **self.schedule_args,
    )


jax.tree_util.register_pytree_node_class(AnnealingScheduleConfig)
