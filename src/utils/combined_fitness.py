from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jaxtyping import PRNGKeyArray, PyTree

from src.scoring.cai import cai_score
from src.scoring.mpnn import mpnn_score

from .nucleotide import translate
from .types import (
  EvoSequence,
  FitnessWeights,
  MPNNModel,
  PopulationSequenceBools,
  PopulationSequenceFloats,
  PopulationSequences,
)


@partial(jit)
def combine_fitness_scores(
  fitness_components: dict,
  fitness_weights: Optional[FitnessWeights] = None,
) -> PopulationSequenceFloats:
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
    combined_fitness = jnp.sum(jnp.array(list(fitness_components.values())), axis=0)

  return combined_fitness


@partial(
  jit,
  static_argnames=(
    "protein_length",
    "fitness_combine_func",
    "fitness_flags",
    "mpnn_model_instance",
    "fitness_funcs",  # Mark fitness_funcs as a static argument
    "mpnn_model_is_active",
  ),
)
def calculate_fitness_population(
  key: PRNGKeyArray,
  sequences: PopulationSequences,
  fitness_funcs: tuple[Callable, ...],
  fitness_combine_func: Callable,
  protein_length: int,
  fitness_flags: list[bool],
  mpnn_model_instance: Optional[MPNNModel] = None,
  mpnn_model_is_active: bool = False,
  fitness_weights: Optional[FitnessWeights] = None,
) -> tuple[PopulationSequenceFloats, PyTree, PopulationSequenceBools]:
  """Calculates fitness for a population of particles using configurable functions.

  Args:
      key: PRNG key for random operations.
      sequences: Population of sequences to evaluate.
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
      - Valid translation flags
  """
  num_particles = sequences.shape[0]
  # Create a new set of PRNG keys for each particle in the population
  keys_for_vmap = jax.random.split(key, num_particles)

  # Helper function to calculate fitness for a single sequence
  def calculate_fitness_single(key: PRNGKeyArray, seq: EvoSequence) -> tuple:
    nuc_seq = False
    aa_seq = False
    valid_translation = jnp.array(False, dtype=jnp.bool_)
    translated = None
    if seq.shape[0] == protein_length * 3:
      nuc_seq = True
      translated, valid_translation = translate(seq)
      aa_seq = True
    elif seq.shape[0] == protein_length:
      translated = seq
      valid_translation = jnp.all(seq != -1)
      aa_seq = True
    else:
      raise ValueError(
        "Invalid sequence length. " "Expected either protein_length * 3 or protein_length."
      )
    fitness_components = {}

    for i, (func, is_active) in enumerate(zip(fitness_funcs, fitness_flags)):
      if is_active:
        if func.__name__ == "mpnn_score" and aa_seq:
          assert mpnn_model_instance is not None, "MPNN model required"
          assert translated is not None, "Translated sequence is required"
          fitness_components[f"fitness_{i}"] = mpnn_score(translated, key, mpnn_model_instance)
        elif func.__name__ == "cai_score" and nuc_seq:
          fitness_components[f"fitness_{i}"] = cai_score(seq, translated)
        else:
          fitness_components[f"fitness_{i}"] = func(key, seq, translated)
      else:
        fitness_components[f"fitness_{i}"] = jnp.array(0.0, dtype=jnp.float32)

    if fitness_weights is not None:
      combined_fitness = fitness_combine_func(fitness_components, fitness_weights)
    else:
      combined_fitness = fitness_combine_func(fitness_components)

    return combined_fitness, fitness_components, valid_translation

  # Vectorize the single-particle fitness calculation function
  vmapped_fitness_func = vmap(
    calculate_fitness_single,
    in_axes=(0, 0),
  )

  # Apply the vectorized function to the population
  return vmapped_fitness_func(keys_for_vmap, sequences)
