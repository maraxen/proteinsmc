from logging import getLogger

import jax.numpy as jnp
from jax import jit, vmap

from .types import BatchSequences, ScalarFloat

logger = getLogger(__name__)

@jit
def calculate_position_entropy(pos_seqs: BatchSequences) -> ScalarFloat:
  _, counts = jnp.unique(pos_seqs, return_counts=True)
  probs = counts / counts.sum()
  return -jnp.sum(probs * jnp.log(probs))

@jit
def shannon_entropy(seqs: BatchSequences) -> ScalarFloat:
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
        logger.warning(
            f"Warning: shannon_entropy received unexpected type {type(seqs)}"
        )
        return jnp.array(0.0, dtype=jnp.float32)
