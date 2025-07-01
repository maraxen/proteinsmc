from functools import partial

import jax.numpy as jnp
from jax import jit, lax, random, vmap
from jaxtyping import (
    Array,
    Bool,
    Float,
    Int,
    PRNGKeyArray,
)

from .constants import (
    CODON_INT_TO_RES_INT_JAX,
    COLABDESIGN_X_INT,
)

# Type aliases for common shape descriptors
# These tell the linter that these are intentional names for shape annotations
NucleotideSequence = Int[Array, "nuc_len"]
ProteinSequence = Int[Array, "protein_len"]
ParticlesBatch = Int[Array, "n_particles nuc_len"]
BatchProteinSequences = Int[Array, "n_particles protein_len"]
ScalarFloat = Float[Array, ""]
ScalarBool = Bool[Array, ""]


@partial(jit, static_argnames=('n_nuc_alphabet_size', 'protein_length', 'mu_nuc')) 
def initial_mutation_kernel_no_x_jax(
    key: PRNGKeyArray,
    particles_template_batch: ParticlesBatch,
    mu_nuc: float,
    n_nuc_alphabet_size: int,
    protein_length: int
) -> ParticlesBatch:
    """
    Applies random nucleotide mutations to template sequences, ensuring no 'X'
    codons are introduced by these initial mutations. If a mutation would cause
    an 'X', that codon is reverted to its state in the template.
    
    Args:
        key: JAX PRNG key.
        particles_template_batch: JAX array of nucleotide sequences 
                                  (shape: (n_particles, nuc_len)).
        mu_nuc: Mutation rate for nucleotides.
        n_nuc_alphabet_size: Size of the nucleotide alphabet 
                             (e.g., 4 for A, C, G, T).
        protein_length: Length of the protein sequence (number of amino acids).
        
    Returns:
        final_particles_batch: JAX array of nucleotide sequences after mutation
                               and 'X' check.
    """
    n_particles, n_nuc_total = particles_template_batch.shape

    # 1. Propose nucleotide mutations (standard method)
    key_mask, key_offsets = random.split(key)
    mutation_mask_attempt = random.uniform(
        key_mask, shape=particles_template_batch.shape
    ) < mu_nuc

    offsets = random.randint(
        key_offsets,
        shape=particles_template_batch.shape,
        minval=1,
        maxval=n_nuc_alphabet_size
    )
    proposed_nucleotides = (
        particles_template_batch + offsets
    ) % n_nuc_alphabet_size

    # Particles with all mutations applied, some might have created 'X' codons
    particles_with_all_proposed_mutations = jnp.where(
        mutation_mask_attempt,
        proposed_nucleotides,
        particles_template_batch
    )

    # 2. Define a per-particle function to check and revert X-causing codons
    def check_and_revert_x_codons_single_particle(
        template_particle_nucs: NucleotideSequence,
        proposed_particle_nucs: NucleotideSequence
    ) -> NucleotideSequence: 
        # Initialize the final nucleotide sequence for this particle with the
        # proposed mutations
        final_particle_nucs = proposed_particle_nucs.astype(jnp.int32)

        # Iterate over codons (this loop will be unrolled by JAX due to static
        # protein_length)
        for i in range(protein_length):
            codon_start_idx = i * 3

            # Get the original codon from the template (before any mutation in
            # this step)
            original_codon = lax.dynamic_slice_in_dim(
                template_particle_nucs, codon_start_idx, 3
            )

            # Get the codon as it is after all proposed nucleotide mutations
            proposed_codon = lax.dynamic_slice_in_dim(
                proposed_particle_nucs, codon_start_idx, 3
            )

            # Translate the proposed_codon
            # Ensure indices are int32 for JAX array indexing if necessary,
            # though usually fine if source is int32
            n1, n2, n3 = proposed_codon[0], proposed_codon[1], proposed_codon[2]
            translated_aa_proposed_codon = CODON_INT_TO_RES_INT_JAX[n1, n2, n3]

            # Check if this specific codon was actually changed by a mutation
            # AND resulted in 'X'
            codon_was_mutated = jnp.any(original_codon != proposed_codon)
            proposed_codon_is_x = (
                translated_aa_proposed_codon == COLABDESIGN_X_INT
            )

            # If a mutation occurred in this codon and it resulted in 'X',
            # revert this codon to its original (template) state.
            revert_this_codon = codon_was_mutated & proposed_codon_is_x

            current_codon_to_keep = jnp.where(
                revert_this_codon, original_codon, proposed_codon
            )

            # Update the final nucleotide sequence for this codon
            final_particle_nucs = lax.dynamic_update_slice(
                final_particle_nucs,
                current_codon_to_keep,
                [codon_start_idx]
            )

        return final_particle_nucs

    # Apply the check and revert logic to each particle in the batch
    final_particles_batch = vmap(check_and_revert_x_codons_single_particle)(
        particles_template_batch,
        particles_with_all_proposed_mutations
    )
    final_particles_batch = final_particles_batch.astype(jnp.int32)

    return final_particles_batch