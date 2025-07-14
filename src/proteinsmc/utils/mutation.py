"""Utilities for mutating sequences."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
from jax import jit, random, vmap

if TYPE_CHECKING:
  from jaxtyping import PRNGKeyArray

  from proteinsmc.utils.types import NucleotideSequence, PopulationSequences

from .constants import (
  CODON_INT_TO_RES_INT_JAX,
  COLABDESIGN_X_INT,
)
from .vmap_utils import chunked_vmap


@partial(jit, static_argnames=("n_states", "mutation_rate"))
def mutate(
  key: PRNGKeyArray,
  sequences: PopulationSequences,
  mutation_rate: float,
  n_states: int,
) -> PopulationSequences:
  """Apply random mutations to a population of nucleotide sequences.

  Args:
      key: JAX PRNG key.
      sequences: JAX array of nucleotide sequences
                  (shape: (n, nuc_len)).
      mutation_rate: Mutation rate for nucleotides.
      n_states: Number of states. e.g., nucleotide types (4 for A, C, G, T).

  Returns: JAX array of mutated nucleotide sequences.

  """
  key_mutate, key_offsets = random.split(key)
  mutation_mask = random.uniform(key_mutate, shape=sequences.shape) < mutation_rate
  offsets = random.randint(key_offsets, shape=sequences.shape, minval=1, maxval=n_states)
  proposed_mutations = (sequences + offsets) % n_states
  mutated_sequences = jnp.where(mutation_mask, proposed_mutations, sequences)
  return mutated_sequences.astype(jnp.int8)


@jit
def _revert_x_codons_if_mutated(
  template_nucleotide_sequences: NucleotideSequence,
  candidate_nucleotide_sequences: NucleotideSequence,
) -> NucleotideSequence:
  """Revert codons in the proposed nucleotide sequence that were mutated and resulted in 'X'.

  Args:
      template_nucleotide_sequences: JAX array of nucleotide sequence from the template
                              (shape: (nuc_len,)).
      candidate_nucleotide_sequences: JAX array of nucleotide sequence after mutations
                              (shape: (nuc_len,)).
      protein_length: Length of the protein sequence (number of amino acids).

  Returns:
      final_nucleotide_sequences: JAX array of nucleotide sequence after reverting
                            'X' codons (shape: (nuc_len,)).

  """
  final_nucleotide_sequences = candidate_nucleotide_sequences.astype(jnp.int8)
  codons_template = template_nucleotide_sequences.reshape(-1, 3)
  codons_candidate = candidate_nucleotide_sequences.reshape(-1, 3)

  n1 = codons_candidate[:, 0]
  n2 = codons_candidate[:, 1]
  n3 = codons_candidate[:, 2]
  translated_aa = CODON_INT_TO_RES_INT_JAX[n1, n2, n3]

  codon_was_mutated = jnp.any(codons_template != codons_candidate, axis=1)
  codon_is_x = translated_aa == COLABDESIGN_X_INT
  revert_mask = codon_was_mutated & codon_is_x

  codons_final = jnp.where(revert_mask[:, None], codons_template, codons_candidate)

  final_nucleotide_sequences = codons_final.reshape(-1)

  return final_nucleotide_sequences.astype(jnp.int8)


@partial(jit, static_argnames=("n_states",))
def diversify_initial_nucleotide_sequences(
  key: PRNGKeyArray,
  seed_sequences: PopulationSequences,
  mutation_rate: float,
  n_states: int,
) -> PopulationSequences:
  """Apply random nucleotide mutations to template sequences, ensuring no 'X' codons are introduced.

  If a mutation would cause an 'X', that codon is reverted to its state in the template.

  Args:
      key: JAX PRNG key.
      seed_sequences: JAX array of sequences
                                (shape: (population_size, sequence_length)).
      mutation_rate: Mutation rate for nucleotides.
      n_states: Number of states (e.g., 4 for A, C, G, T in nucleotides).

  Returns:
      final_population: JAX array of nucleotide sequences after mutation
                            and 'X' check.

  """
  key_mask, key_offsets = random.split(key)
  mutation_mask_attempt = random.uniform(key_mask, shape=seed_sequences.shape) < mutation_rate

  offsets = random.randint(key_offsets, shape=seed_sequences.shape, minval=1, maxval=n_states)
  proposed_nucleotides = (seed_sequences + offsets) % n_states

  particles_with_all_proposed_mutations = jnp.where(
    mutation_mask_attempt,
    proposed_nucleotides,
    seed_sequences,
  )

  final_population = vmap(_revert_x_codons_if_mutated)(
    seed_sequences,
    particles_with_all_proposed_mutations,
  )
  final_population = final_population.astype(jnp.int8)

  return final_population.astype(jnp.int8)


@partial(jit, static_argnames=("n_states",))
def diversify_initial_protein_sequences(
  key: PRNGKeyArray,
  seed_sequences: PopulationSequences,
  mutation_rate: float,
  n_states: int,
) -> PopulationSequences:
  """Apply random protein mutations to template sequences.

  Args:
      key: JAX PRNG key.
      seed_sequences: JAX array of sequences
                                (shape: (population_size, sequence_length)).
      mutation_rate: Mutation rate for proteins.
      n_states: Number of states (e.g., 20 for amino acids).

  Returns:
      final_population: JAX array of protein sequences after mutation.

  """
  key_mask, key_offsets = random.split(key)
  mutation_mask_attempt = random.uniform(key_mask, shape=seed_sequences.shape) < mutation_rate

  offsets = random.randint(key_offsets, shape=seed_sequences.shape, minval=1, maxval=n_states)
  proposed_amino_acids = (seed_sequences + offsets) % n_states

  final_population = jnp.where(
    mutation_mask_attempt,
    proposed_amino_acids,
    seed_sequences,
  )

  return final_population.astype(jnp.int8)


def diversify_initial_sequences(
  key: PRNGKeyArray,
  seed_sequences: PopulationSequences,
  mutation_rate: float,
  sequence_type: Literal["nucleotide", "protein"],
) -> PopulationSequences:
  """Diversify initial sequences based on the sequence type."""
  if sequence_type == "nucleotide":
    return diversify_initial_nucleotide_sequences(
      key,
      seed_sequences,
      mutation_rate,
      n_states=4,
    )
  if sequence_type == "protein":
    return diversify_initial_protein_sequences(
      key,
      seed_sequences,
      mutation_rate,
      n_states=20,
    )
  msg = f"Unsupported sequence_type='{sequence_type}'"
  raise ValueError(msg)


@partial(jit, static_argnames=("n_states", "mutation_rate"))
def mutate_single(
  key: PRNGKeyArray,
  sequence: NucleotideSequence,
  mutation_rate: float,
  n_states: int,
) -> NucleotideSequence:
  """Apply random mutations to a single nucleotide sequence.

  Args:
      key: JAX PRNG key.
      sequence: JAX array of nucleotide sequence (shape: (nuc_len,)).
      mutation_rate: Mutation rate for nucleotides.
      n_states: Number of states. e.g., nucleotide types (4 for A, C, G, T).

  Returns: JAX array of mutated nucleotide sequence.

  """
  key_mutate, key_offsets = random.split(key)
  mutation_mask = random.uniform(key_mutate, shape=sequence.shape) < mutation_rate
  offsets = random.randint(key_offsets, shape=sequence.shape, minval=1, maxval=n_states)
  proposed_mutations = (sequence + offsets) % n_states
  mutated_sequence = jnp.where(mutation_mask, proposed_mutations, sequence)
  return mutated_sequence.astype(jnp.int8)


def chunked_mutation_step(
  key: PRNGKeyArray,
  population: PopulationSequences,
  mutation_rate: float,
  n_states: int,
  chunk_size: int,
) -> PopulationSequences:
  """Apply mutation to population using chunked vmap processing.

  This function demonstrates how to use `chunked_vmap` with a function
  that operates on a single item (`mutate_single`).

  Args:
    key: PRNG key for mutation.
    population: Population to mutate.
    mutation_rate: Rate of mutation.
    n_states: Number of states (e.g., 20 for amino acids).
    chunk_size: Size of chunks for processing.

  Returns:
    Mutated population.

  """

  def fn(
    k: PRNGKeyArray,
    seq: NucleotideSequence,
  ) -> NucleotideSequence:
    """Mutate a single sequence."""
    return mutate_single(k, seq, mutation_rate, n_states)

  mutation_keys = random.split(key, population.shape[0])

  mutated_population = chunked_vmap(
    func=fn,
    data=(mutation_keys, population),
    chunk_size=chunk_size,
    in_axes=(0, 0),
    static_args=None,
  )
  return mutated_population.astype(jnp.int8)


def dispatch_mutation_single(
  key: PRNGKeyArray,
  sequence: NucleotideSequence,
  mutation_rate: float,
  sequence_type: Literal["nucleotide", "protein"],
) -> NucleotideSequence:
  """Dispatches the appropriate sequence processing function for a single sequence."""
  if sequence_type == "nucleotide":
    n_states = 4
    return mutate_single(key, sequence, mutation_rate, n_states)
  if sequence_type == "protein":
    n_states = 20
    return mutate_single(key, sequence, mutation_rate, n_states)
  msg = f"Unsupported sequence_type='{sequence_type}'"
  raise ValueError(msg)


def dispatch_mutation(
  key: PRNGKeyArray,
  sequences: PopulationSequences,
  mutation_rate: float,
  sequence_type: Literal["nucleotide", "protein"],
) -> PopulationSequences:
  """Dispatches the appropriate sequence processing function based on the sequence type."""
  if sequence_type == "nucleotide":
    n_states = 4
    return mutate(key, sequences, mutation_rate, n_states)
  if sequence_type == "protein":
    n_states = 20
    return mutate(key, sequences, mutation_rate, n_states)
  msg = f"Unsupported sequence_type='{sequence_type}'"
  raise ValueError(msg)
