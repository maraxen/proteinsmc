from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jaxtyping import PRNGKeyArray, PyTree

from src.mpnn import mpnn_score

from .nucleotide import cai_score, translate
from .types import (
    BatchSequenceBools,
    BatchSequenceFloats,
    BatchSequences,
    FitnessWeights,
    MCSequence,
    MPNNModel,
)


@partial(jit, static_argnames=('fitness_weights',))
def combine_fitness_scores(
    fitness_components: dict,
    fitness_weights: Optional[FitnessWeights] = None
) -> BatchSequenceFloats:
    """Combines individual fitness scores into a single score using weights.
    
    Args:
        fitness_components: Dictionary of individual fitness scores.
        fitness_weights: Optional weights for combining fitness scores.
        
    Returns:
        Combined fitness score.
    """
    if fitness_weights is not None:
        combined_fitness = jnp.tensordot(
            fitness_weights, jnp.array(list(fitness_components.values())), axes=1
        )
    else:
        combined_fitness = jnp.sum(
            jnp.array(list(fitness_components.values())), axis=0
        )
    
    return combined_fitness

@partial(
    jit, 
    static_argnames=(
        'protein_length', 'fitness_combine_func', 
        'fitness_flags', 'mpnn_model_instance'
    )
)
def calculate_fitness_batch(
  key: PRNGKeyArray,
  sequences: BatchSequences,
  fitness_funcs: tuple[Callable, ...],
  fitness_combine_func: Callable,
  protein_length: int,
  fitness_flags: list[bool],
  mpnn_model_instance: Optional[MPNNModel] = None,
  fitness_weights: Optional[FitnessWeights] = None,
) -> tuple[BatchSequenceFloats, PyTree, BatchSequenceBools]:
  """Calculates fitness for a batch of particles using configurable functions.
  
  Args:
    key: PRNG key for random operations.
    nucleotide_sequences: Batch of nucleotide sequences to evaluate.
    fitness_funcs: List of fitness function callables.
    fitness_combine_func: Function to combine individual fitness scores.
    protein_length: Length of the protein sequence.
    fitness_flags: Boolean flags indicating which fitness functions to use.
    mpnn_model_instance: Optional MPNN model instance.
    fitness_weights: Optional weights for combining fitness scores.
    
  Returns:
    Tuple containing:
    - Combined fitness scores for each sequence
    - Individual fitness components (as a PyTree)
    - Translated protein sequences
    - Valid translation flags
  """
  num_particles = sequences.shape[0]
  # Create a new set of PRNG keys for each particle in the batch
  keys_for_vmap = jax.random.split(key, num_particles)

  # Helper function to calculate fitness for a single sequence
  def calculate_fitness_single(key: PRNGKeyArray, seq: MCSequence) -> tuple:
    # Translate nucleotides to amino acids
    nuc_seq = False
    aa_seq = False
    valid_translation = jnp.array(False, dtype=jnp.bool_)
    translated = None
    if seq.shape[0] != protein_length * 3:
      nuc_seq = True
      translated = translate(seq)
      valid_translation = ~jnp.any(translated == -1)
      aa_seq = True
    elif seq.shape[0] == protein_length:
      translated = seq
      valid_translation = jnp.all(seq != -1)
      aa_seq = True
    else:
      raise ValueError(
          "Invalid sequence length. "
          "Expected either protein_length * 3 or protein_length."
      )
    fitness_components = {} #TODO: have this more structured for jax
  
    # Calculate each fitness component based on flags
    for i, (func, is_active) in enumerate(zip(fitness_funcs, fitness_flags)):
      if is_active:
        if func.__name__ == 'mpnn_score' and aa_seq:
          assert mpnn_model_instance is not None, "MPNN model required"
          assert translated is not None, "Translated sequence is required"
          fitness_components[f'fitness_{i}'] = mpnn_score(
              key, translated, mpnn_model_instance
          )
        elif func.__name__ == 'cai_score' and nuc_seq:
          fitness_components[f'fitness_{i}'] = cai_score(nuc_seq, translated)
        else:
          fitness_components[f'fitness_{i}'] = func(key, nuc_seq, translated)
      else:
        # If function is inactive, provide zero score
        fitness_components[f'fitness_{i}'] = jnp.array(0.0, dtype=jnp.float32)
    
    # Combine fitness scores using the provided combination function
    if fitness_weights is not None:
      combined_fitness = fitness_combine_func(
          fitness_components, fitness_weights
      )
    else:
      combined_fitness = fitness_combine_func(fitness_components)
    
    # Handle invalid translations by setting fitness to -infinity
    combined_fitness = jnp.where(
        valid_translation, combined_fitness, -jnp.inf
    )
    
    return combined_fitness, fitness_components, valid_translation

  # Vectorize the single-particle fitness calculation function
  vmapped_fitness_func = vmap(
    calculate_fitness_single,
    in_axes=(0, 0),
  )
  
  # Apply the vectorized function to the batch
  return vmapped_fitness_func(keys_for_vmap, sequences)

