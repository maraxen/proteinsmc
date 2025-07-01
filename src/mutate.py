from functools import partial

import jax.numpy as jnp
from jax import jit, random
from jaxtyping import PRNGKeyArray

from .utils.types import BatchSequences


@partial(jit, static_argnames=('N_alphabet', 'mu'))
def mutation_kernel(
    key: PRNGKeyArray, 
    sequences: BatchSequences, 
    mu: float, 
    N_alphabet: int
) -> BatchSequences:
    """Applies random mutations to a batch of nucleotide sequences.
    Args:
        key: JAX PRNG key.
        sequences: JAX array of sequences (shape: (n_particles, seq_len)).
        mu: Mutation rate for nucleotides.
        N_alphabet: Number of types types (e.g., 4 for A, C, G, T) or 20 for
                    amino acids.
    Returns: JAX array of mutated sequences."""
    key_mutate, key_offsets = random.split(key)
    # Create a boolean mask indicating which nucleotides to mutate
    mutation_mask = random.uniform(key_mutate, shape=sequences.shape) < mu
    # Generate random offsets (1 to N_alphabet-1) to change nucleotide type
    offsets = random.randint(
        key_offsets, shape=sequences.shape, minval=1, maxval=N_alphabet
    )
    # Propose new nucleotides: (current_nucleotide + offset) % N_alphabet
    mutated_nucleotides_proposed = (sequences + offsets) % N_alphabet
    # Apply mutations where the mask is True
    mutated_nucleotides = jnp.where(
        mutation_mask, mutated_nucleotides_proposed, sequences
    )
    return mutated_nucleotides
