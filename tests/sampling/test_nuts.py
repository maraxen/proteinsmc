import jax.numpy as jnp
import pytest
from jax import random

from proteinsmc.sampling.nuts import nuts_sampler


def log_prob_fn(x):
  return -0.5 * jnp.sum(x**2)


@pytest.mark.parametrize(
  "num_samples, warmup_steps, num_chains",
  [(100, 50, 1), (200, 100, 4), (500, 200, 2)],
)
def test_nuts_sampler_output_shape(num_samples, warmup_steps, num_chains):
  key = random.PRNGKey(0)
  initial_position = jnp.zeros(2)

  samples = nuts_sampler(
    key,
    log_prob_fn,
    initial_position,
    num_samples=num_samples,
    warmup_steps=warmup_steps,
    num_chains=num_chains,
  )

  assert samples.shape == (num_chains, num_samples, 2)


def test_nuts_sampler_convergence():
  key = random.PRNGKey(42)
  dim = 2
  initial_position = jnp.zeros(dim)
  num_samples = 2000
  warmup_steps = 500
  num_chains = 4

  samples = nuts_sampler(
    key,
    log_prob_fn,
    initial_position,
    num_samples=num_samples,
    warmup_steps=warmup_steps,
    num_chains=num_chains,
  )
  assert jnp.allclose(jnp.mean(samples), 0.0, atol=0.1)
  assert jnp.allclose(jnp.var(samples), 1.0, atol=0.5)


@pytest.mark.parametrize("step_size, adapt_step_size", [(0.1, False), (1.0, True)])
def test_nuts_sampler_options(step_size, adapt_step_size):
  key = random.PRNGKey(0)
  initial_position = jnp.zeros(1)
  num_samples = 100
  warmup_steps = 50

  samples = nuts_sampler(
    key,
    log_prob_fn,
    initial_position,
    num_samples=num_samples,
    warmup_steps=warmup_steps,
    step_size=step_size,
    adapt_step_size=adapt_step_size,
  )

  assert samples.shape == (1, num_samples, 1)


def test_nuts_sampler_high_dimension():
  dim = 10
  key = random.PRNGKey(123)
  initial_position = random.normal(key, (dim,))
  num_samples = 100
  warmup_steps = 50

  samples = nuts_sampler(key, log_prob_fn, initial_position, num_samples, warmup_steps)

  assert samples.shape == (1, num_samples, dim)
