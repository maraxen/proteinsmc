import jax.numpy as jnp

from src.utils.annealing_schedules import (
  cosine_schedule,
  exponential_schedule,
  linear_schedule,
  static_schedule,
)

TOL = 1e-6


def test_linear_schedule():
  assert linear_schedule(p=jnp.array(1), n_steps=jnp.array(10), beta_max=jnp.array(1.0)) == 0.0
  assert linear_schedule(p=jnp.array(10), n_steps=jnp.array(10), beta_max=jnp.array(1.0)) == 1.0
  assert linear_schedule(p=jnp.array(11), n_steps=jnp.array(10), beta_max=jnp.array(1.0)) == 1.0
  assert jnp.allclose(
    linear_schedule(p=jnp.array(5), n_steps=jnp.array(10), beta_max=jnp.array(1.0)),
    4.0 / 9.0,
    atol=TOL,
  )


def test_exponential_schedule():
  assert (
    exponential_schedule(
      p=jnp.array(1),
      n_steps=jnp.array(10),
      beta_max=jnp.array(1.0),
    )
    == 0.0
  )
  assert (
    exponential_schedule(
      p=jnp.array(10),
      n_steps=jnp.array(10),
      beta_max=jnp.array(1.0),
    )
    == 1.0
  )
  assert (
    exponential_schedule(
      p=jnp.array(11),
      n_steps=jnp.array(10),
      beta_max=jnp.array(1.0),
    )
    == 1.0
  )
  # Test intermediate value
  p = 5
  n_steps = 10
  beta_max = 1.0
  rate = 5.0
  x = (p - 1) / (n_steps - 1)
  expected = beta_max * (jnp.exp(rate * x) - 1) / (jnp.exp(rate) - 1)
  assert jnp.allclose(
    exponential_schedule(
      p=jnp.array(p),
      n_steps=jnp.array(n_steps),
      beta_max=jnp.array(beta_max),
      rate=jnp.array(rate),
    ),
    expected,
    atol=TOL,
  )


def test_cosine_schedule():
  assert cosine_schedule(p=jnp.array(1), n_steps=jnp.array(10), beta_max=jnp.array(1.0)) == 0.0
  assert cosine_schedule(p=jnp.array(10), n_steps=jnp.array(10), beta_max=jnp.array(1.0)) == 1.0
  assert cosine_schedule(p=jnp.array(11), n_steps=jnp.array(10), beta_max=jnp.array(1.0)) == 1.0
  assert jnp.allclose(
    cosine_schedule(p=jnp.array(5), n_steps=jnp.array(10), beta_max=jnp.array(1.0)),
    0.5 * (1.0 - jnp.cos(jnp.pi * 4.0 / 9.0)),
    atol=TOL,
  )


def test_static_schedule():
  assert static_schedule(beta_max=jnp.array(0.5)) == 0.5
  assert static_schedule(beta_max=jnp.array(1.0)) == 1.0
