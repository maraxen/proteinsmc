from typing import Callable

import jax
import jax.numpy as jnp
from colabdesign.mpnn.model import mk_mpnn_model
from jax import random

from ..experiment import calculate_cai_single_jax as calculate_cai_jax
from ..experiment import translate_single_jax as nucleotide_to_amino_acid_jax
from ..mpnn import mpnn_score as calculate_mpnn_score_jax
from .constants import NUCLEOTIDES_CHAR, NUCLEOTIDES_INT_MAP, RES_TO_CODON_CHAR
from .helper_functions import (
    initial_mutation_kernel_no_x_jax,
)

# --- Constants (moved from constants.py if not already there) ---
# Assuming NUCLEOTIDES_CHAR, NUCLEOTIDES_INT_MAP, RES_TO_CODON_CHAR are already in constants.py
# If not, they should be moved here or imported from constants.py

# --- Domain-Specific Fitness Calculation (moved from smc.py) ---
def calculate_fitness_batch_jax(
    key: jax.Array,
    particles_batch: jax.Array,
    sequence_type: str,
    protein_length: int,
    mpnn_model_instance: mk_mpnn_model,
    mpnn_model_is_active_static: bool
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    if sequence_type == "protein":
        # Convert nucleotide sequences to amino acid sequences
        aa_seqs_int, has_x_flags = nucleotide_to_amino_acid_jax(
            particles_batch, protein_length
        )

        # Calculate CAI (Codon Adaptation Index)
        cai_values = calculate_cai_jax(particles_batch, protein_length)

        # Calculate MPNN scores
        if mpnn_model_is_active_static:
            mpnn_scores = calculate_mpnn_score_jax(
                key, aa_seqs_int, mpnn_model_instance
            )
        else:
            mpnn_scores = jnp.zeros(particles_batch.shape[0])

        # Combine fitness components (example: simple sum or weighted sum)
        # This is a placeholder; actual combination logic might be more complex
        combined_fitness = cai_values + mpnn_scores # Example combination

        return combined_fitness, cai_values, mpnn_scores, aa_seqs_int, has_x_flags
    elif sequence_type == "nucleotide":
        raise NotImplementedError("Nucleotide-specific fitness calculation not yet implemented.")
    else:
        raise ValueError(f"Unsupported sequence_type: {sequence_type}")

# --- Domain-Specific Initial Population Generation ---
def get_protein_nucleotide_initial_population_fn(
    initial_aa_seq_char: str,
    protein_length: int,
    initial_population_mutation_rate: float
) -> Callable[[jax.Array, int], jax.Array]:
    N_nuc_total = 3 * protein_length
    initial_nucleotide_seq_int_list = [NUCLEOTIDES_INT_MAP['A']] * N_nuc_total
    try:
        for i in range(protein_length):
            aa_char = initial_aa_seq_char[i]
            codon_char_list = list(RES_TO_CODON_CHAR[aa_char])
            for j in range(3):
                initial_nucleotide_seq_int_list[3*i + j] = \
                    NUCLEOTIDES_INT_MAP[codon_char_list[j]]
    except KeyError as e:
        raise ValueError(
            f"Failed to generate initial JAX nucleotide template from AA '{e}'. "
            f"Check RES_TO_CODON_CHAR and initial_aa_seq_char."
        ) from e
    initial_nucleotide_template_one_seq_jax = jnp.array(
        initial_nucleotide_seq_int_list, dtype=jnp.int32
    )

    def initial_population_fn(key: jax.Array, n_particles: int) -> jax.Array:
        _particles_jax_template_batch = jnp.tile(
            initial_nucleotide_template_one_seq_jax, (n_particles, 1)
        )
        particles_jax_initial = initial_mutation_kernel_no_x_jax(
            key=key,
            particles_template_batch=_particles_jax_template_batch,
            mu_nuc=initial_population_mutation_rate,
            n_nuc_alphabet_size=len(NUCLEOTIDES_CHAR),
            protein_length=protein_length # This implies nucleotide length
        )
        return particles_jax_initial
    return initial_population_fn

# --- Domain-Specific Fitness Function Factories ---
def get_protein_nucleotide_fitness_fns(
    mpnn_model_instance: mk_mpnn_model,
    mpnn_model_is_active_static: bool,
    protein_length: int
) -> tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:

    def batch_fitness_fn(particles_batch: jax.Array) -> jax.Array:
        # Assuming calculate_fitness_batch_jax returns (fitness_values, ...)
        key_for_fitness, _ = random.split(random.PRNGKey(0)) # Dummy key, actual key from SMC loop
        fitness_values, _, _, _, _ = calculate_fitness_batch_jax(
            key_for_fitness,  # This key needs to be managed carefully if used
            # inside a jitted function
            # inside a jitted function
            particles_batch,
            protein_length,
            mpnn_model_instance,
            mpnn_model_is_active_static
        )
        return fitness_values

    def single_fitness_fn(particle: jax.Array) -> jax.Array:
        # Wrap batch_fitness_fn for single particle
        key_for_fitness, _ = random.split(random.PRNGKey(0)) # Dummy key
        fitness_value, _, _, _, _ = calculate_fitness_batch_jax(
            key_for_fitness, # This key needs to be managed carefully
            jnp.expand_dims(particle, axis=0), # Make it a batch of 1
            protein_length,
            mpnn_model_instance,
            mpnn_model_is_active_static
        )
        return fitness_value[0] # Return scalar fitness

    return batch_fitness_fn, single_fitness_fn

# --- Domain-Specific Entropy Calculation (moved from smc.py) ---
# These functions are already generic enough, but their usage is domain-specific
# They are kept here for clarity that they are part of the protein/nucleotide domain
# calculate_amino_acid_entropy_py
# calculate_nucleotide_entropy_py
