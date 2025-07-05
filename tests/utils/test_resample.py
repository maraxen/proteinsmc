import jax
import jax.numpy as jnp
import pytest

from proteinsmc.utils.resample import resample


@pytest.fixture
def simple_particles():
  return jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])


def test_resample_shapes(simple_particles):
  key = jax.random.PRNGKey(0)
  log_weights = jnp.log(jnp.array([0.25, 0.25, 0.25, 0.25]))
  n_particles = simple_particles.shape[0]

  resampled_particles, ess, normalized_weights = resample(key, simple_particles, log_weights)

  assert resampled_particles.shape == simple_particles.shape
  assert isinstance(ess, jax.Array)
  assert ess.shape == ()
  assert normalized_weights.shape == (n_particles,)


def test_resample_uniform_weights(simple_particles):
  key = jax.random.PRNGKey(0)
  log_weights = jnp.zeros(simple_particles.shape[0])  # Uniform weights

  resampled_particles, ess, normalized_weights = resample(key, simple_particles, log_weights)

  assert jnp.allclose(ess, simple_particles.shape[0])
  assert jnp.allclose(
    normalized_weights, jnp.full_like(log_weights, 1.0 / simple_particles.shape[0])
  )


def test_resample_skewed_weights(simple_particles):
  key = jax.random.PRNGKey(0)
  # All weight on the last particle
  log_weights = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, 0.0])

  resampled_particles, ess, normalized_weights = resample(key, simple_particles, log_weights)

  assert jnp.allclose(ess, 1.0)
  expected_weights = jnp.array([0.0, 0.0, 0.0, 1.0])
  assert jnp.allclose(normalized_weights, expected_weights, atol=1e-7)
  # All resampled particles should be the last particle
  assert jnp.all(resampled_particles == simple_particles[-1])


def test_resample_inf_weights(simple_particles):
  key = jax.random.PRNGKey(0)
  log_weights = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf, -jnp.inf])

  resampled_particles, ess, normalized_weights = resample(key, simple_particles, log_weights)

  # The implementation handles -inf by replacing with a large negative number,
  # so softmax should still produce uniform weights.
  assert jnp.allclose(ess, simple_particles.shape[0])
  assert jnp.allclose(
    normalized_weights, jnp.full_like(log_weights, 1.0 / simple_particles.shape[0])
  )
