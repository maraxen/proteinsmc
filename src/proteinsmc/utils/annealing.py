"""Defines various annealing schedules for training."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax import jit
from jaxtyping import Float

if TYPE_CHECKING:
  from jaxtyping import Array

  from proteinsmc.models.annealing import (
    AnnealingConfig,
    AnnealingFn,
  )

CurrentBetaFloat = Float[jax.Array, ""]

ANNEALING_REGISTRY = {}


def register_schedule(
  name: str,
) -> Callable[[AnnealingFn], AnnealingFn]:
  """Register an annealing schedule function."""

  def decorator(func: AnnealingFn) -> AnnealingFn:
    """Decorate to register an annealing schedule function."""
    ANNEALING_REGISTRY[name] = func
    return func

  return decorator


def get_annealing_function(config: AnnealingConfig) -> AnnealingFn:
  """Get a configured annealing schedule function."""
  if config.annealing_fn not in ANNEALING_REGISTRY:
    msg = f"Unknown annealing schedule: {config.annealing_fn}"
    raise ValueError(msg)

  func = ANNEALING_REGISTRY[config.annealing_fn]

  @jit
  def annealing_fn(current_step: int, _context: Array | None = None) -> float:
    """JIT-compatible annealing function."""
    return func(
      current_step=current_step,
      n_steps=config.n_steps,
      beta_min=config.beta_min,
      beta_max=config.beta_max,
      _context=_context,
      **config.kwargs,
    )

  return annealing_fn


@register_schedule("linear")
@partial(jit, static_argnames=("_context",))
def linear_schedule(
  current_step: int,
  n_steps: int,
  beta_min: float,
  beta_max: float,
  _context: Array | None = None,
) -> CurrentBetaFloat:
  """Linear annealing schedule for beta that is JAX-compatible.

  Args:
    current_step (int): The current annealing step (1-based).
    n_steps (int): Total number of annealing steps.
    beta_min (float): Minimum beta value.
    beta_max (float): Maximum beta value.
    _context (Array | None): Optional context for schedule.

  Returns:
    CurrentBetaFloat: The annealed beta value for the current step.

  Raises:
    TypeError: If the output is not a JAX array.

  Example:
    >>> linear_schedule(1, 10, 0.1, 1.0)
    Array(0.1, dtype=float32)

  """
  step_val = beta_min + (current_step - 1) / (n_steps - 1) * (beta_max - beta_min)
  result = jnp.where(current_step >= n_steps, beta_max, step_val)
  return jnp.clip(result, beta_min, beta_max)


@register_schedule("exponential")
@partial(jit, static_argnames=("_context", "rate"))
def exponential_schedule(
  current_step: int,
  n_steps: int,
  beta_min: float,
  beta_max: float,
  _context: Array | None = None,
  rate: float = 5.0,
) -> CurrentBetaFloat:
  """Exponential annealing schedule for beta that starts at beta_min and ends at beta_max.

  Args:
    current_step (int): The current annealing step (1-based).
    n_steps (int): Total number of annealing steps.
    beta_min (float): Minimum beta value.
    beta_max (float): Maximum beta value.
    _context (Array | None): Optional context for schedule.
    rate (float): Exponential growth rate.

  Returns:
    CurrentBetaFloat: The annealed beta value for the current step.

  Raises:
    TypeError: If the output is not a JAX array.

  Example:
    >>> exponential_schedule(1, 10, 0.1, 1.0)
    Array(0.1, dtype=float32)

  """
  if current_step <= 1:
    return jnp.array(beta_min, dtype=jnp.float32)
  if current_step >= n_steps:
    return jnp.array(beta_max, dtype=jnp.float32)
  x = (current_step - 1) / (n_steps - 1)
  exp_val = jnp.exp(jnp.minimum(rate * x, 700.0))
  denominator = jnp.exp(rate) - 1.0
  eps = 1e-9
  if abs(denominator) < eps:
    scale_factor = (beta_max - beta_min) / (denominator + eps)
  else:
    scale_factor = (beta_max - beta_min) / denominator
  result = beta_min + scale_factor * (exp_val - 1)
  return jnp.clip(result, beta_min, beta_max)


@register_schedule("cosine")
@partial(jit, static_argnames=("_context",))
def cosine_schedule(
  current_step: int,
  n_steps: int,
  beta_min: float,
  beta_max: float,
  _context: Array | None = None,
) -> CurrentBetaFloat:
  """Cosine annealing schedule for beta that starts at beta_min and ends at beta_max.

  Args:
    current_step (int): The current annealing step (1-based).
    n_steps (int): Total number of annealing steps.
    beta_min (float): Minimum beta value.
    beta_max (float): Maximum beta value.
    _context (Array | None): Optional context for schedule.

  Returns:
    CurrentBetaFloat: The annealed beta value for the current step.

  Raises:
    TypeError: If the output is not a JAX array.

  Example:
    >>> cosine_schedule(1, 10, 0.1, 1.0)
    Array(0.1, dtype=float32)

  """
  if current_step <= 1:
    return jnp.array(beta_min, dtype=jnp.float32)
  if current_step >= n_steps:
    return jnp.array(beta_max, dtype=jnp.float32)
  x = (current_step - 1) / (n_steps - 1)
  result = beta_min + 0.5 * (beta_max - beta_min) * (1.0 - jnp.cos(jnp.pi * x))
  return jnp.clip(result, beta_min, beta_max)


@register_schedule("static")
@partial(jit, static_argnames=("_context",))
def static_schedule(
  current_step: int,
  _n_steps: int,
  beta_min: float,
  beta_max: float,
  _context: Array | None = None,
) -> CurrentBetaFloat:
  """Implement static annealing schedule, constant at beta max.

  Args:
    current_step (int): The current annealing step (1-based).
    n_steps (int): Total number of annealing steps.
    beta_min (float): Minimum beta value.
    beta_max (float): Maximum beta value.
    _context (Array | None): Optional context for schedule.

  Returns:
    CurrentBetaFloat: The annealed beta value for the current step.

  Raises:
    TypeError: If the output is not a JAX array.

  Example:
    >>> static_schedule(1, 10, 0.1, 1.0)
    Array(0.1, dtype=float32)

  """
  if current_step <= 1:
    return jnp.array(beta_min, dtype=jnp.float32)
  return jnp.array(beta_max, dtype=jnp.float32)
