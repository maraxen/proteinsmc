"""Utilities for mutating sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
from jax import jit, random, vmap

if TYPE_CHECKING:
  from jaxtyping import Float, Int, PRNGKeyArray

  from proteinsmc.models.types import EvoSequence, NucleotideSequence

from .constants import (
  CODON_INT_TO_RES_INT_JAX,
  PROTEINMPNN_X_INT,
)
from .jax_utils import chunked_map


def mutate(
  key: PRNGKeyArray,
  sequence: EvoSequence,
  mutation_rate: Float,
  q_states: Int,
) -> EvoSequence:
  """Apply random mutations to a population of nucleotide sequences.

  Args:
      key: JAX PRNG key.
      sequence: JAX array of nucleotide sequences
                  (shape: (n, nuc_len)).
      mutation_rate: Mutation rate for nucleotides.
      q_states: Number of states. e.g., nucleotide types (4 for A, C, G, T).

  Returns: JAX array of mutated nucleotide sequences.

  """
  key_mutate, key_offsets = random.split(key)
  mutation_mask = random.uniform(key_mutate, shape=sequence.shape) < mutation_rate
  offsets = random.randint(key_offsets, shape=sequence.shape, minval=1, maxval=q_states)
  proposed_mutations = (sequence + offsets) % q_states
  return jnp.asarray(jnp.where(mutation_mask, proposed_mutations, sequence), dtype=jnp.int8)


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
  codon_is_x = translated_aa == PROTEINMPNN_X_INT
  revert_mask = codon_was_mutated & codon_is_x

  codons_final = jnp.where(revert_mask[:, None], codons_template, codons_candidate)

  final_nucleotide_sequences = codons_final.reshape(-1)

  return final_nucleotide_sequences.astype(jnp.int8)


def diversify_initial_nucleotide_sequences(
  key: PRNGKeyArray,
  seed_sequences: EvoSequence,
  mutation_rate: Float,
) -> EvoSequence:
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

  offsets = random.randint(key_offsets, shape=seed_sequences.shape, minval=0, maxval=4)
  proposed_nucleotides = (seed_sequences + offsets) % 4

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


def diversify_initial_protein_sequences(
  key: PRNGKeyArray,
  seed_sequences: EvoSequence,
  mutation_rate: Float,
) -> EvoSequence:
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

  offsets = random.randint(key_offsets, shape=seed_sequences.shape, minval=0, maxval=20)
  proposed_amino_acids = (seed_sequences + offsets) % 20

  final_population = jnp.where(
    mutation_mask_attempt,
    proposed_amino_acids,
    seed_sequences,
  )

  return final_population.astype(jnp.int8)


def diversify_initial_sequences(
  key: PRNGKeyArray,
  seed_sequences: EvoSequence,
  mutation_rate: Float,
  sequence_type: Literal["nucleotide", "protein"],
) -> EvoSequence:
  """Diversify initial sequences based on the sequence type."""
  if sequence_type == "nucleotide":
    return diversify_initial_nucleotide_sequences(
      key,
      seed_sequences,
      mutation_rate,
    )
  if sequence_type == "protein":
    return diversify_initial_protein_sequences(
      key,
      seed_sequences,
      mutation_rate,
    )
  msg = f"Unsupported sequence_type='{sequence_type}'"
  raise ValueError(msg)


def chunked_mutation_step(
  key: PRNGKeyArray,
  population: EvoSequence,
  mutation_rate: float,
  n_states: int,
  chunk_size: int,
) -> EvoSequence:
  """Apply mutation to population using chunked map processing.

  This function demonstrates how to use `chunked_map` with a function
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
    return mutate(k, seq, mutation_rate, n_states)

  mutation_keys = random.split(key, population.shape[0])

  mutated_population = chunked_map(
    func=fn,
    data=(mutation_keys, population),
    chunk_size=chunk_size,
  )
  return mutated_population.astype(jnp.int8)
