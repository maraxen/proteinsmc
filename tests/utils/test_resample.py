
import jax
import jax.numpy as jnp
import chex
import pytest

from proteinsmc.utils.resampling import resample


@pytest.fixture
def simple_particles():
  return jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])


def test_resample_shapes(simple_particles):
  key = jax.random.PRNGKey(0)
  log_weights = jnp.log(jnp.array([0.25, 0.25, 0.25, 0.25]))
  n_particles = simple_particles.shape[0]

  resampled_particles, ess, normalized_weights = resample(key, simple_particles, log_weights)

  chex.assert_shape(resampled_particles, simple_particles.shape)
  assert isinstance(ess, jax.Array)
  chex.assert_shape(ess, ())
  chex.assert_shape(normalized_weights, (n_particles,))


def test_resample_uniform_weights(simple_particles):
  key = jax.random.PRNGKey(0)
  log_weights = jnp.zeros(simple_particles.shape[0])  # Uniform weights

  resampled_particles, ess, normalized_weights = resample(key, simple_particles, log_weights)

  chex.assert_trees_all_close(ess, simple_particles.shape[0])
  chex.assert_trees_all_close(
    normalized_weights, jnp.full_like(log_weights, 1.0 / simple_particles.shape[0])
  )


def test_resample_skewed_weights(simple_particles):
  key = jax.random.PRNGKey(0)
  log_weights = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, 0.0])

  resampled_particles, ess, normalized_weights = resample(key, simple_particles, log_weights)

  chex.assert_trees_all_close(ess, 1.0)
  expected_weights = jnp.array([0.0, 0.0, 0.0, 1.0])
  chex.assert_trees_all_close(normalized_weights, expected_weights, atol=1e-7)
  chex.assert_trees_all_equal(
    resampled_particles,
    jnp.tile(simple_particles[-1], (simple_particles.shape[0], 1)),
  )


def test_resample_inf_weights(simple_particles):
  key = jax.random.PRNGKey(0)
  log_weights = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf])

  resampled_particles, ess, normalized_weights = resample(key, simple_particles, log_weights)

  chex.assert_trees_all_close(ess, simple_particles.shape[0])
  chex.assert_trees_all_close(
    normalized_weights, jnp.full_like(log_weights, 1.0 / simple_particles.shape[0])
  )
