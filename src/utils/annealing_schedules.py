import jax.numpy as jnp

from .types import (
  ScalarFloat,
  ScalarInt,
)


def linear_schedule(p: ScalarInt, n_steps: ScalarInt, beta_max: ScalarFloat) -> ScalarFloat:
  """Linear annealing schedule for beta."""
  if p <= 1:
    return jnp.array(0.0, dtype=jnp.float32)
  if p >= n_steps:
    return jnp.array(beta_max, dtype=jnp.float32)
  return jnp.array((beta_max * (p - 1) / (n_steps - 1)), dtype=jnp.float32)


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
  denominator = jnp.exp(rate) - 1.0  # Corrected denominator
  if abs(denominator) < 1e-9:
    scale_factor = beta_max / (denominator + 1e-9)
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


def static_schedule(beta_max: ScalarFloat | float) -> ScalarFloat:
  """Static annealing schedule (beta is constant)."""
  return jnp.array(beta_max, dtype=jnp.float32)
