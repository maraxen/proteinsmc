"""Defines various annealing schedules for training."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import jit

if TYPE_CHECKING:
  from jaxtyping import Array, Float, Int

from proteinsmc.models.annealing import (
  AnnealingRegistryItem,
  AnnealingScheduleRegistry,
  CurrentBetaFloat,
)


@partial(jit, static_argnames=("_context",))
def linear_schedule(
  current_step: Int,
  n_steps: Int,
  beta_max: Float,
  _context: Array | None = None,
) -> CurrentBetaFloat:
  """Linear annealing schedule for beta that is JAX-compatible."""
  step_val = (current_step - 1) / (n_steps - 1) * beta_max
  result = jnp.where(current_step >= n_steps, beta_max, step_val)
  result = jnp.where(current_step <= 1, 0.0, result)
  if not isinstance(result, jnp.ndarray):
    msg = f"Expected result to be a JAX array, got {type(result)}"
    raise TypeError(msg)
  return jnp.where(current_step <= 1, 0.0, result)


default_rate = jnp.array(5.0, dtype=jnp.float32)


@partial(jit, static_argnames=("_context", "rate"))
def exponential_schedule(
  current_step: Int,
  n_steps: Int,
  beta_max: Float,
  _context: Array | None = None,
  rate: Float = default_rate,
) -> CurrentBetaFloat:
  """Exponential annealing schedule for beta."""
  if current_step <= 1:
    return jnp.array(0.0, dtype=jnp.float32)
  if current_step >= n_steps:
    return jnp.array(beta_max, dtype=jnp.float32)
  x = (current_step - 1) / (n_steps - 1)
  exp_val = jnp.exp(jnp.minimum(rate * x, 700.0))
  denominator = jnp.exp(rate) - 1.0
  eps = 1e-9
  if abs(denominator) < eps:
    scale_factor = beta_max / (denominator + eps)
  else:
    scale_factor = beta_max / denominator
  return scale_factor * (exp_val - 1)


@partial(jit, static_argnames=("_context",))
def cosine_schedule(
  current_step: Int,
  n_steps: Int,
  beta_max: Float,
  _context: Array | None = None,
) -> CurrentBetaFloat:
  """Cosine annealing schedule for beta."""
  if current_step <= 1:
    return jnp.array(0.0, dtype=jnp.float32)
  if current_step >= n_steps:
    return jnp.array(beta_max, dtype=jnp.float32)
  x = (current_step - 1) / (n_steps - 1)
  return beta_max * 0.5 * (1.0 - jnp.cos(jnp.pi * x))


@partial(jit, static_argnames=("_context",))
def static_schedule(
  current_step: Int,
  n_steps: Int,
  beta_max: Float,
  _context: Array | None = None,
) -> CurrentBetaFloat:
  """Run static annealing schedule (beta is constant)."""
  if current_step <= 1:
    return jnp.array(0.0, dtype=jnp.float32)
  if current_step >= n_steps:
    return jnp.array(beta_max, dtype=jnp.float32)
  return jnp.array(beta_max, dtype=jnp.float32)


ANNEALING_REGISTRY = AnnealingScheduleRegistry(
  items={
    "linear": AnnealingRegistryItem(lambda: linear_schedule, name="linear"),
    "exponential": AnnealingRegistryItem(
      lambda: exponential_schedule,
    ),
    "cosine": AnnealingRegistryItem(
      lambda: cosine_schedule,
      name="cosine",
    ),
    "static": AnnealingRegistryItem(
      lambda: static_schedule,
      name="static",
    ),
  },
)
