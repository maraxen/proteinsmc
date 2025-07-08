"""Defines various annealing schedules for training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from proteinsmc.utils.types import (
    ScalarFloat,
    ScalarInt,
  )


@dataclass(frozen=True)
class AnnealingScheduleConfig:
  """Configuration for an annealing schedule.

  Attributes:
      schedule_fn: Callable function that defines the annealing schedule.
      beta_max: Maximum value for beta.
      annealing_len: Number of steps over which to anneal.
      schedule_args: Additional arguments for the schedule function.

  """

  schedule_fn: Callable[[ScalarInt, ScalarInt, ScalarFloat], ScalarFloat]
  beta_max: float
  annealing_len: int
  schedule_args: tuple = field(default_factory=tuple)

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children = ()
    aux_data = {k: v for k, v in self.__dict__.items() if k not in children}
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data: dict, _children: tuple) -> AnnealingScheduleConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(**aux_data)


jax.tree_util.register_pytree_node_class(AnnealingScheduleConfig)


def linear_schedule(p: ScalarInt, n_steps: ScalarInt, beta_max: ScalarFloat) -> ScalarFloat:
  """Linear annealing schedule for beta that is JAX-compatible."""
  step_val = (p - 1) / (n_steps - 1) * beta_max
  result = jnp.where(p >= n_steps, beta_max, step_val)
  return jnp.where(p <= 1, 0.0, result)


default_rate = jnp.array(5.0, dtype=jnp.float32)


def exponential_schedule(
  p: ScalarInt,
  n_steps: ScalarInt,
  beta_max: ScalarFloat,
  rate: ScalarFloat = default_rate,
) -> ScalarFloat:
  """Exponential annealing schedule for beta."""
  if p <= 1:
    return jnp.array(0.0, dtype=jnp.float32)
  if p >= n_steps:
    return jnp.array(beta_max, dtype=jnp.float32)
  x = (p - 1) / (n_steps - 1)
  exp_val = jnp.exp(jnp.minimum(rate * x, 700.0))
  denominator = jnp.exp(rate) - 1.0
  eps = 1e-9
  if abs(denominator) < eps:
    scale_factor = beta_max / (denominator + eps)
  else:
    scale_factor = beta_max / denominator
  return scale_factor * (exp_val - 1)


def cosine_schedule(p: ScalarInt, n_steps: ScalarInt, beta_max: ScalarFloat) -> ScalarFloat:
  """Cosine annealing schedule for beta."""
  if p <= 1:
    return jnp.array(0.0, dtype=jnp.float32)
  if p >= n_steps:
    return jnp.array(beta_max, dtype=jnp.float32)
  x = (p - 1) / (n_steps - 1)
  return beta_max * 0.5 * (1.0 - jnp.cos(jnp.pi * x))


def static_schedule(_p: ScalarInt, _n: ScalarInt, beta_max: ScalarFloat) -> ScalarFloat:
  """Run static annealing schedule (beta is constant)."""
  return jnp.array(beta_max, dtype=jnp.float32)
