import jax
import jax.numpy as jnp
from jax import jit, random
from jaxtyping import PRNGKeyArray

from .types import PopulationSequenceFloats, PopulationSequences, ScalarFloat


@jit
def resample(
  key: PRNGKeyArray,
  particles: PopulationSequences,
  log_weights: PopulationSequenceFloats,
) -> tuple[PopulationSequences, ScalarFloat, PopulationSequenceFloats]:
  """
  Performs systematic resampling on a population of particles.
  Args:
      key: JAX PRNG key.
      particles: JAX array of particles (shape: (n_particles, seq_len)).
      log_weights: JAX array of log weights for each particle.
      n_particles: Number of particles (static for JIT).
  Returns:
        - Resampled particles.
        - Effective Sample Size (ESS).
        - Normalized weights.
  """
  n_particles = particles.shape[0]
  log_weights_safe = jnp.where(jnp.isneginf(log_weights), -1e9, log_weights)
  normalized_weights = jax.nn.softmax(log_weights_safe)
  ess = 1.0 / jnp.sum(jnp.square(normalized_weights))
  u = random.uniform(key, (n_particles,))
  cumulative_weights = jnp.cumsum(normalized_weights)
  indices = jnp.searchsorted(cumulative_weights, u)
  resampled_particles = particles[indices]
  return resampled_particles, ess, normalized_weights
