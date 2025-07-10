import chex

import jax.numpy as jnp

from proteinsmc.utils.annealing_schedules import (
  cosine_schedule,
  exponential_schedule,
  linear_schedule,
  static_schedule,
)

TOL = 1e-6


def test_linear_schedule():
  chex.assert_trees_all_equal(
    linear_schedule(p=jnp.array(1), n_steps=jnp.array(10), beta_max=jnp.array(1.0)),
    0.0,
  )
  chex.assert_trees_all_equal(
    linear_schedule(p=jnp.array(10), n_steps=jnp.array(10), beta_max=jnp.array(1.0)),
    1.0,
  )
  chex.assert_trees_all_equal(
    linear_schedule(p=jnp.array(11), n_steps=jnp.array(10), beta_max=jnp.array(1.0)),
    1.0,
  )
  chex.assert_trees_all_close(
    linear_schedule(p=jnp.array(5), n_steps=jnp.array(10), beta_max=jnp.array(1.0)),
    4.0 / 9.0,
    atol=TOL,
  )


def test_exponential_schedule():
  chex.assert_trees_all_equal(
    exponential_schedule(
      p=jnp.array(1),
      n_steps=jnp.array(10),
      beta_max=jnp.array(1.0),
    ),
    0.0,
  )
  chex.assert_trees_all_equal(
    exponential_schedule(
      p=jnp.array(10),
      n_steps=jnp.array(10),
      beta_max=jnp.array(1.0),
    ),
    1.0,
  )
  chex.assert_trees_all_equal(
    exponential_schedule(
      p=jnp.array(11),
      n_steps=jnp.array(10),
      beta_max=jnp.array(1.0),
    ),
    1.0,
  )
  p = 5
  n_steps = 10
  beta_max = 1.0
  rate = 5.0
  x = (p - 1) / (n_steps - 1)
  expected = beta_max * (jnp.exp(rate * x) - 1) / (jnp.exp(rate) - 1)
  chex.assert_trees_all_close(
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
  chex.assert_trees_all_equal(
    cosine_schedule(p=jnp.array(1), n_steps=jnp.array(10), beta_max=jnp.array(1.0)),
    0.0,
  )
  chex.assert_trees_all_equal(
    cosine_schedule(p=jnp.array(10), n_steps=jnp.array(10), beta_max=jnp.array(1.0)),
    1.0,
  )
  chex.assert_trees_all_equal(
    cosine_schedule(p=jnp.array(11), n_steps=jnp.array(10), beta_max=jnp.array(1.0)),
    1.0,
  )
  chex.assert_trees_all_close(
    cosine_schedule(p=jnp.array(5), n_steps=jnp.array(10), beta_max=jnp.array(1.0)),
    0.5 * (1.0 - jnp.cos(jnp.pi * 4.0 / 9.0)),
    atol=TOL,
  )


def test_static_schedule():
  chex.assert_trees_all_equal(
    static_schedule(
      _p=jnp.array(1, dtype=jnp.int32),
      _n=jnp.array(10, dtype=jnp.int32),
      beta_max=jnp.array(0.5),
    ),
    0.5,
  )
  chex.assert_trees_all_equal(
    static_schedule(
      _p=jnp.array(1, dtype=jnp.int32),
      _n=jnp.array(10, dtype=jnp.int32),
      beta_max=jnp.array(1.0),
    ),
    1.0,
  )
