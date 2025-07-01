"""
Protein SMC Experiment core logic, converted from the provided Jupyter cell.
All constants, mappings, and main experiment functions are defined here.
"""
# --- Imports ---
import math  # noqa: F401
import random  # noqa: F401
from functools import partial  # noqa: F401

import colabdesign  # noqa: F401
import jax  # noqa: F401
import jax.numpy as jnp
import numpy as np  # noqa: F401
import pyarrow as pa  # noqa: F401
import pyarrow.parquet as pq  # noqa: F401
from jax import jit
from scipy.special import logsumexp as scipy_logsumexp  # noqa: F401

from .utils.constants import (
    CODON_INT_TO_RES_INT_JAX,
    COLABDESIGN_X_INT,
    ECOLI_CODON_FREQ_JAX,
    ECOLI_MAX_FREQS_JAX,
)


# --- JAX Helper Functions ---
@jit
def translate_single_jax(nucleotide_seq_int):
    """Translates a nucleotide sequence to an amino acid sequence.
    
    Uses ColabDesign's AA integers.
    """
    protein_len = nucleotide_seq_int.shape[0] // 3
    codons_int = nucleotide_seq_int[:protein_len*3].reshape((protein_len, 3))
    aa_seq_int = CODON_INT_TO_RES_INT_JAX[
        codons_int[:,0], codons_int[:,1], codons_int[:,2]
    ]
    has_x_residue = jnp.any(aa_seq_int == COLABDESIGN_X_INT)
    return aa_seq_int, has_x_residue

@jit
def calculate_cai_single_jax(nucleotide_seq_int, aa_seq_int_for_cai):
    """Calculates Codon Adaptation Index (CAI).
    
    `aa_seq_int_for_cai` uses ColabDesign's AA integers.
    """
    protein_len = nucleotide_seq_int.shape[0] // 3
    codons_int = nucleotide_seq_int[:protein_len*3].reshape((protein_len, 3))
    codon_frequencies = ECOLI_CODON_FREQ_JAX[
        codons_int[:,0], codons_int[:,1], codons_int[:,2]
    ]
    max_aa_frequencies = ECOLI_MAX_FREQS_JAX[aa_seq_int_for_cai]
    wi = codon_frequencies / jnp.maximum(max_aa_frequencies, 1e-9)
    valid_codon_mask = (aa_seq_int_for_cai != COLABDESIGN_X_INT)
    log_wi = jnp.log(jnp.maximum(wi, 1e-12))
    sum_log_wi = jnp.sum(log_wi * valid_codon_mask)
    num_valid_codons = jnp.sum(valid_codon_mask)
    cai = jnp.exp(sum_log_wi / jnp.maximum(num_valid_codons, 1.0))
    return jnp.where(num_valid_codons > 0, cai, 0.0)

# ...continue porting kernels and experiment logic as functions...

def run_experiment():
    print("Protein SMC experiment would run here. (Stub)")
    # TODO: Port all logic from the Jupyter cell into this module as
    #       functions/classes. This includes all constants, mappings,
    #       kernels, and the main experiment loop.
