from functools import partial
from typing import Literal

import jax.numpy as jnp
from jax import jit, lax, random, vmap
from jaxtyping import (
  PRNGKeyArray,
)

from .constants import (
  CODON_INT_TO_RES_INT_JAX,
  COLABDESIGN_X_INT,
)
from .types import NucleotideSequence, PopulationSequences
from .translate import translate


@partial(jit, static_argnames=("n_states", "mutation_rate"))
def mutate(
  key: PRNGKeyArray, sequences: PopulationSequences, mutation_rate: float, n_states: int
) -> PopulationSequences:
  """Applies random mutations to a population of nucleotide sequences.
  Args:
      key: JAX PRNG key.
      sequences: JAX array of nucleotide sequences
                  (shape: (n_particles, nuc_len)).
      mutation_rate: Mutation rate for nucleotides.
      n_states: Number of states. e.g., nucleotide types (4 for A, C, G, T).

  Returns: JAX array of mutated nucleotide sequences.
  """
  key_mutate, key_offsets = random.split(key)
  mutation_mask = random.uniform(key_mutate, shape=sequences.shape) < mutation_rate
  offsets = random.randint(key_offsets, shape=sequences.shape, minval=1, maxval=n_states)
  proposed_mutations = (sequences + offsets) % n_states
  mutated_sequences = jnp.where(mutation_mask, proposed_mutations, sequences)
  return mutated_sequences


@partial(jit, static_argnames=("sequence_length",))
def _revert_x_codons_if_mutated(
  template_nucleotide_sequences: NucleotideSequence,
  candidate_nucleotide_sequences: NucleotideSequence,
  sequence_length: int,
) -> NucleotideSequence:
  """Reverts codons in the proposed nucleotide sequence that were mutated
  and resulted in 'X'
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
  final_nucleotide_sequences = candidate_nucleotide_sequences.astype(jnp.int32)

  for i in range(sequence_length):
    codon_start_idx = i * 3
    original_codon = lax.dynamic_slice_in_dim(template_nucleotide_sequences, codon_start_idx, 3)
    proposed_codon = lax.dynamic_slice_in_dim(candidate_nucleotide_sequences, codon_start_idx, 3)
    n1, n2, n3 = proposed_codon[0], proposed_codon[1], proposed_codon[2]
    translated_aa_proposed_codon = CODON_INT_TO_RES_INT_JAX[n1, n2, n3]
    codon_was_mutated = jnp.any(original_codon != proposed_codon)
    proposed_codon_is_x = translated_aa_proposed_codon == COLABDESIGN_X_INT
    revert_this_codon = codon_was_mutated & proposed_codon_is_x
    current_codon_to_keep = jnp.where(revert_this_codon, original_codon, proposed_codon)
    final_nucleotide_sequences = lax.dynamic_update_slice(
      final_nucleotide_sequences, current_codon_to_keep, [codon_start_idx]
    )

  return final_nucleotide_sequences


@partial(jit, static_argnames=("n_nuc_alphabet_size", "protein_length", "mu_nuc"))
def diversify_initial_sequences(
  key: PRNGKeyArray,
  template_sequences: PopulationSequences,
  mutation_rate: float,
  n_states: int,
  sequence_length: int,
  nucleotide: bool = True,
) -> PopulationSequences:
  """
  Applies random nucleotide mutations to template sequences, ensuring no 'X'
  codons are introduced by these initial mutations. If a mutation would cause
  an 'X', that codon is reverted to its state in the template.

  Args:
      key: JAX PRNG key.
      template_sequences: JAX array of sequences
                                (shape: (population_size, sequence_length)).
      mutation_rate: Mutation rate for nucleotides.
      n_states: Number of states (e.g., 4 for A, C, G, T in nucleotides).
      sequence_length: Length of the protein sequence (number of amino acids).
      nucleotide: Whether to treat the sequences as nucleotide sequences.

  Returns:
      final_particles_population: JAX array of nucleotide sequences after mutation
                            and 'X' check.
  """
  sequence_length = template_sequences.shape[1]

  key_mask, key_offsets = random.split(key)
  mutation_mask_attempt = random.uniform(key_mask, shape=template_sequences.shape) < mutation_rate

  offsets = random.randint(key_offsets, shape=template_sequences.shape, minval=1, maxval=n_states)
  proposed_nucleotides = (template_sequences + offsets) % n_states

  particles_with_all_proposed_mutations = jnp.where(
    mutation_mask_attempt, proposed_nucleotides, template_sequences
  )

  if nucleotide:
    final_particles_population = vmap(
      partial(_revert_x_codons_if_mutated, sequence_length=sequence_length)
    )(template_sequences, particles_with_all_proposed_mutations)
    final_particles_population = final_particles_population.astype(jnp.int32)
  else:
    final_particles_population = particles_with_all_proposed_mutations

  return final_particles_population


def dispatch_mutation(
  key: PRNGKeyArray,
  sequences: PopulationSequences,
  mutation_rate: float,
  sequence_type: Literal["nucleotide", "protein"],
  evolve_as: Literal["nucleotide", "protein"],
) -> PopulationSequences:
  """
  Dispatches the appropriate sequence processing function based on the sequence type.
  Args:
      key: JAX PRNG key.
      sequences: JAX array of nucleotide sequences
                  (shape: (n_particles, nuc_len)).
      mutation_rate: Mutation rate for nucleotides.
      n_states: Number of states. e.g., nucleotide types (4 for A, C, G, T).
      sequence_length: Length of the protein sequence (number of amino acids).
  """
  if sequence_type == "nucleotide" and evolve_as == "nucleotide":
    n_states = 4
    return mutate(key, sequences, mutation_rate, n_states)
  elif sequence_type == "protein" and evolve_as == "protein":
    n_states = 20
    return mutate(key, sequences, mutation_rate, n_states)
  elif sequence_type == "nucleotide" and evolve_as == "protein":
    n_states = 20
    sequences = translate(sequences, n_states)
    return mutate(key, sequences, mutation_rate, n_states)
  else:
    raise ValueError(f"Unsupported sequence type: {sequence_type}")
