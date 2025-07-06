from functools import partial
from typing import Literal

import jax.numpy as jnp
from jax import jit, random, vmap
from jaxtyping import (
  PRNGKeyArray,
)

from .constants import (
  CODON_INT_TO_RES_INT_JAX,
  COLABDESIGN_X_INT,
)
from .types import NucleotideSequence, PopulationSequences


@partial(jit, static_argnames=("n_states", "mutation_rate"))
def mutate(
  key: PRNGKeyArray, sequences: PopulationSequences, mutation_rate: float, n_states: int
) -> PopulationSequences:
  """Applies random mutations to a population of nucleotide sequences.
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


@partial(jit, static_argnames=("n_states", "nucleotide"))
def diversify_initial_sequences(
  key: PRNGKeyArray,
  template_sequences: PopulationSequences,
  mutation_rate: float,
  n_states: int,
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
      final_population: JAX array of nucleotide sequences after mutation
                            and 'X' check.
  """
  key_mask, key_offsets = random.split(key)
  mutation_mask_attempt = random.uniform(key_mask, shape=template_sequences.shape) < mutation_rate

  offsets = random.randint(key_offsets, shape=template_sequences.shape, minval=1, maxval=n_states)
  proposed_nucleotides = (template_sequences + offsets) % n_states

  particles_with_all_proposed_mutations = jnp.where(
    mutation_mask_attempt, proposed_nucleotides, template_sequences
  )

  if nucleotide:
    final_population = vmap(_revert_x_codons_if_mutated)(
      template_sequences, particles_with_all_proposed_mutations
    )
    final_population = final_population.astype(jnp.int8)
  else:
    final_population = particles_with_all_proposed_mutations.astype(jnp.int8)

  return final_population.astype(jnp.int8)


def dispatch_mutation(
  key: PRNGKeyArray,
  sequences: PopulationSequences,
  mutation_rate: float,
  sequence_type: Literal["nucleotide", "protein"],
) -> PopulationSequences:
  """
  Dispatches the appropriate sequence processing function based on the sequence type.
  """
  if sequence_type == "nucleotide":
    n_states = 4
    return mutate(key, sequences, mutation_rate, n_states)
  elif sequence_type == "protein":
    n_states = 20
    return mutate(key, sequences, mutation_rate, n_states)
  else:
    raise ValueError(f"Unsupported sequence_type='{sequence_type}'")
