from logging import getLogger

import jax.numpy as jnp
from jax import jit, vmap

from .types import PopulationSequenceFloats, PopulationSequences, ScalarFloat

logger = getLogger(__name__)


@jit
def calculate_logZ_increment(
  log_weights: PopulationSequenceFloats,
  population_size: int,
) -> ScalarFloat:
  """Calculate log evidence increment from log weights.

  Args:
    log_weights: Log weights for the population
    population_size: Size of the population

  Returns:
    Log evidence increment for this step
  """
  if log_weights.shape[0] == 0:
    return jnp.ndarray(-jnp.inf)

  valid_log_weights = jnp.where(jnp.isneginf(log_weights), -jnp.inf, log_weights)
  max_l_w = jnp.max(valid_log_weights)
  safe_max_l_w = jnp.where(jnp.isneginf(max_l_w), 0.0, max_l_w)
  log_sum_exp_weights = safe_max_l_w + jnp.log(jnp.sum(jnp.exp(valid_log_weights - safe_max_l_w)))

  current_logZ_increment = log_sum_exp_weights - jnp.log(jnp.maximum(population_size, 1.0))
  current_logZ_increment = jnp.where(
    jnp.isneginf(log_sum_exp_weights) | (population_size == 0),
    -jnp.inf,
    current_logZ_increment,
  )

  return current_logZ_increment


@jit
def calculate_position_entropy(pos_seqs: PopulationSequences) -> ScalarFloat:
  _, counts = jnp.unique(pos_seqs, return_counts=True, size=pos_seqs.shape[0])
  probs = counts / counts.sum()
  return -jnp.sum(probs * jnp.log(probs + 1e-9))


@jit
def shannon_entropy(seqs: PopulationSequences) -> ScalarFloat:
  """Calculates average per-position Shannon entropy of sequences.
  Args:
      seqs: JAX array of nucleotide or amino sequences (integer encoded).
  Returns:
    Scalar representing the Shannon entropy of the sequences.
  """
  if not seqs.size:
    return jnp.array(0.0, dtype=jnp.float32)
  elif isinstance(seqs, jnp.ndarray):
    n_total = seqs.shape[0]
    if n_total == 0:
      return jnp.array(0.0, dtype=jnp.float32)
    seq_length = seqs.shape[1]
    position_entropies = vmap(calculate_position_entropy)(seqs.T)
    entropy = jnp.sum(position_entropies)
    entropy = entropy / seq_length
    return entropy
  else:
    logger.warning(f"Warning: shannon_entropy received unexpected type {type(seqs)}")
    return jnp.array(0.0, dtype=jnp.float32)
