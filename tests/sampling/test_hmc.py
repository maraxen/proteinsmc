
import jax
import jax.numpy as jnp
import chex
import pytest

from proteinsmc.sampling.hmc import HMCSamplerConfig, hmc_sampler


def gaussian_log_prob(x):
  return -0.5 * jnp.sum(x**2)


@pytest.fixture
def sampler_params():
  config = HMCSamplerConfig(
    step_size=0.1,
    num_leapfrog_steps=20,
    num_samples=2000,
    log_prob_fn=gaussian_log_prob,
  )
  return {
    "key": jax.random.PRNGKey(0),
    "initial_position": jnp.array([10.0, 10.0]),  # Start far from the mean
    "config": config,
  }


def test_hmc_sampler_output_shape(sampler_params):
  """Test that the HMC sampler returns samples of the correct shape."""
  samples = hmc_sampler(**sampler_params)
  expected_shape = (sampler_params["config"].num_samples, *sampler_params["initial_position"].shape)
  chex.assert_shape(samples, expected_shape)


def test_hmc_sampler_converges(sampler_params):
  """Test that the HMC sampler converges to the target distribution."""
  samples = hmc_sampler(**sampler_params)

  burn_in = sampler_params["config"].num_samples // 2
  converged_samples = samples[burn_in:]

  mean = jnp.mean(converged_samples, axis=0)
  std_dev = jnp.std(converged_samples, axis=0)

  chex.assert_trees_all_close(mean, 0.0, atol=0.2)
  chex.assert_trees_all_close(std_dev, 1.0, atol=0.2)


def test_hmc_sampler_leapfrog_step():
  """Test a single leapfrog step."""
  # TODO(marielle): Implement this test
  pass
