from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, lax, vmap
from jaxtyping import PRNGKeyArray

from .utils.constants import (
  CODON_INT_TO_RES_INT_JAX,
  COLABDESIGN_X_INT,
)
from .utils.types import NucleotideSequence, PopulationSequences, ProteinSequence


@partial(jit, static_argnames=("n_nuc_alphabet_size", "protein_length", "mu", "N_alphabet"))
def initial_mutation_kernel(
  key: PRNGKeyArray, template: PopulationSequences, mu: float, N_alphabet: int, protein_length: int
) -> PopulationSequences:
  """
  Applies random nucleotide mutations to template sequences, ensuring no 'X'
  codons are introduced by these initial mutations. If a mutation would cause
  an 'X', that codon is reverted to its state in the template.

  Args:
      key: JAX PRNG key.
      particles_template_population: JAX array of nucleotide sequences
                                (shape: (n_particles, nuc_len)).
      mu: Mutation rate for nucleotides.
      N_alphabet: Size of the nucleotide alphabet (e.g., 4 for A, C, G, T).
      protein_length: Length of the protein sequence (number of amino acids).

  Returns:
      final_particles_population: JAX array of nucleotide sequences after mutation
                             and 'X' check.
  """
  _, seq_len = template.shape
  nuc_seq = False
  if seq_len == protein_length * 3:
    # TODO: maybe have some more robust means of checking
    nuc_seq = True

  # 1. Propose nucleotide mutations (standard method)
  key_mask, key_offsets = jax.random.split(key)
  mutation_mask_attempt = jax.random.uniform(key_mask, shape=template.shape) < mu

  offsets = jax.random.randint(key_offsets, shape=template.shape, minval=1, maxval=N_alphabet)
  proposed_mutations = (template + offsets) % N_alphabet

  # Particles with all mutations applied, some might have created 'X' codons
  proteins_with_proposed_mutations = jnp.where(mutation_mask_attempt, proposed_mutations, template)

  # 2. Define a per-particle function to check and revert X-causing codons
  def check_and_revert_x_nuc(
    template_seq: NucleotideSequence, proposed_seq: NucleotideSequence, protein_length: int
  ) -> NucleotideSequence:
    # Initialize the final nucleotide sequence for this particle with the
    # proposed mutations
    final_particle_nucs = proposed_seq
    for i in range(protein_length):
      codon_start_idx = i * 3

      # Get the original codon from the template (before any mutation in
      # this step)
      original_codon = lax.dynamic_slice_in_dim(template_seq, codon_start_idx, 3)

      # Get the codon as it is after all proposed nucleotide mutations
      proposed_codon = lax.dynamic_slice_in_dim(proposed_seq, codon_start_idx, 3)

      # Translate the proposed_codon
      # Ensure indices are int32 for JAX array indexing if necessary,
      # though usually fine if source is int32
      n1, n2, n3 = proposed_codon[0], proposed_codon[1], proposed_codon[2]
      translated_aa_proposed_codon = CODON_INT_TO_RES_INT_JAX[n1, n2, n3]

      # Check if this specific codon was actually changed by a mutation
      # AND resulted in 'X'
      codon_was_mutated = jnp.any(original_codon != proposed_codon)
      proposed_codon_is_x = translated_aa_proposed_codon == COLABDESIGN_X_INT

      # If a mutation occurred in this codon and it resulted in 'X',
      # revert this codon to its original (template) state.
      revert_this_codon = codon_was_mutated & proposed_codon_is_x

      current_codon_to_keep = jnp.where(revert_this_codon, original_codon, proposed_codon)

      # Update the final nucleotide sequence for this codon
      final_particle_nucs = lax.dynamic_update_slice(
        final_particle_nucs, current_codon_to_keep, [codon_start_idx]
      )
    assert isinstance(
      final_particle_nucs, jnp.ndarray
    ), "Final particle nucleotides should be a JAX array."
    return final_particle_nucs

  def check_and_revert_x_aa(
    template_seq: ProteinSequence,
    proposed_seq: ProteinSequence,
  ) -> ProteinSequence:
    x_mask = proposed_seq == COLABDESIGN_X_INT
    reverted_seq = jnp.where(x_mask, template_seq, proposed_seq)
    return reverted_seq

  if nuc_seq:
    # Apply the check and revert logic to each particle in the population
    final_particles_population = vmap(check_and_revert_x_nuc)(
      template, proteins_with_proposed_mutations, protein_length
    )
  else:
    # Apply the check and revert logic to each particle in the population
    final_particles_population = vmap(check_and_revert_x_aa)(
      template, proteins_with_proposed_mutations
    )

  final_particles_population = final_particles_population.astype(jnp.int32)
  return final_particles_population
