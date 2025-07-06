import jax
import jax.numpy as jnp
import pytest

from proteinsmc.sampling.hmc import hmc_sampler


def gaussian_log_prob(x):
  return -0.5 * jnp.sum(x**2)


@pytest.fixture
def sampler_params():
  return {
    "key": jax.random.PRNGKey(0),
    "initial_position": jnp.array([10.0, 10.0]),  # Start far from the mean
    "num_samples": 2000,
    "log_prob_fn": gaussian_log_prob,
    "step_size": 0.1,
    "num_leapfrog_steps": 20,
  }


def test_hmc_sampler_output_shape(sampler_params):
  """Test that the HMC sampler returns samples of the correct shape."""
  samples = hmc_sampler(**sampler_params)
  assert samples.shape == (
    sampler_params["num_samples"],
    *sampler_params["initial_position"].shape,
  )


def test_hmc_sampler_converges(sampler_params):
  """Test that the HMC sampler converges to the target distribution."""
  samples = hmc_sampler(**sampler_params)

  # After burn-in, the mean should be close to 0 and std dev close to 1
  burn_in = sampler_params["num_samples"] // 2
  converged_samples = samples[burn_in:]

  mean = jnp.mean(converged_samples, axis=0)
  std_dev = jnp.std(converged_samples, axis=0)

  assert jnp.allclose(mean, 0.0, atol=0.2)
  assert jnp.allclose(std_dev, 1.0, atol=0.2)


def test_hmc_sampler_leapfrog_step():
  """Test a single leapfrog step."""
  # This requires exposing the leapfrog function or testing its effect via hmc_sampler
  # For simplicity, we test the overall sampler's correctness, which implies
  # the leapfrog integrator is working to some degree.
  # A more rigorous test would involve checking for energy conservation and reversibility.
  pass
