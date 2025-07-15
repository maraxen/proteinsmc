"""Defines various annealing schedules for training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
  from jaxtyping import Float, Int


def linear_schedule(p: Int, n_steps: Int, beta_max: Float) -> Float:
  """Linear annealing schedule for beta that is JAX-compatible."""
  step_val = (p - 1) / (n_steps - 1) * beta_max
  result = jnp.where(p >= n_steps, beta_max, step_val)
  return jnp.where(p <= 1, 0.0, result)


default_rate = jnp.array(5.0, dtype=jnp.float32)


def exponential_schedule(
  p: Int,
  n_steps: Int,
  beta_max: Float,
  rate: Float = default_rate,
) -> Float:
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


def cosine_schedule(p: Int, n_steps: Int, beta_max: Float) -> Float:
  """Cosine annealing schedule for beta."""
  if p <= 1:
    return jnp.array(0.0, dtype=jnp.float32)
  if p >= n_steps:
    return jnp.array(beta_max, dtype=jnp.float32)
  x = (p - 1) / (n_steps - 1)
  return beta_max * 0.5 * (1.0 - jnp.cos(jnp.pi * x))


def static_schedule(_p: Int, _n: Int, beta_max: Float) -> Float:
  """Run static annealing schedule (beta is constant)."""
  return jnp.array(beta_max, dtype=jnp.float32)


ANNEALING_SCHEDULES = {
  "linear": linear_schedule,
  "exponential": exponential_schedule,
  "cosine": cosine_schedule,
  "static": static_schedule,
}
