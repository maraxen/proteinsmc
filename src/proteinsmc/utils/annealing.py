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

  from proteinsmc.models.annealing import AnnealingConfig, AnnealingFuncSignature

CurrentBetaFloat = Float[jax.Array, ""]

ANNEALING_REGISTRY = {}


def register_schedule(
  name: str,
) -> Callable[[AnnealingFuncSignature], AnnealingFuncSignature]:
  """Register an annealing schedule function."""

  def decorator(func: AnnealingFuncSignature):
    ANNEALING_REGISTRY[name] = func
    return func

  return decorator


def get_annealing_function(config: AnnealingConfig) -> AnnealingFuncSignature:
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
      beta_max=config.beta_max,
      _context=_context,
      **config.schedule_args,
    )

  return annealing_fn


@register_schedule("linear")
@partial(jit, static_argnames=("_context",))
def linear_schedule(
  current_step: int,
  n_steps: int,
  beta_max: float,
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


@register_schedule("exponential")
@partial(jit, static_argnames=("_context", "rate"))
def exponential_schedule(
  current_step: int,
  n_steps: int,
  beta_max: float,
  _context: Array | None = None,
  rate: float = 5.0,
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


@register_schedule("cosine")
@partial(jit, static_argnames=("_context",))
def cosine_schedule(
  current_step: int,
  n_steps: int,
  beta_max: float,
  _context: Array | None = None,
) -> CurrentBetaFloat:
  """Cosine annealing schedule for beta."""
  if current_step <= 1:
    return jnp.array(0.0, dtype=jnp.float32)
  if current_step >= n_steps:
    return jnp.array(beta_max, dtype=jnp.float32)
  x = (current_step - 1) / (n_steps - 1)
  return beta_max * 0.5 * (1.0 - jnp.cos(jnp.pi * x))


@register_schedule("static")
@partial(jit, static_argnames=("_context",))
def static_schedule(
  current_step: int,
  n_steps: int,
  beta_max: float,
  _context: Array | None = None,
) -> CurrentBetaFloat:
  """Run static annealing schedule (beta is constant)."""
  if current_step <= 1:
    return jnp.array(0.0, dtype=jnp.float32)
  if current_step >= n_steps:
    return jnp.array(beta_max, dtype=jnp.float32)
  return jnp.array(beta_max, dtype=jnp.float32)
