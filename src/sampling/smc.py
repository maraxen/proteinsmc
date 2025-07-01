from functools import partial
from logging import getLogger
from typing import Callable

import jax
import jax.numpy as jnp
from colabdesign.mpnn.model import mk_mpnn_model
from jax import jit, lax, random

from ..utils.constants import NUCLEOTIDES_CHAR, NUCLEOTIDES_INT_MAP, RES_TO_CODON_CHAR
from ..utils.helper_functions import (
    calculate_amino_acid_entropy_py,
    calculate_nucleotide_entropy_py,
    initial_mutation_kernel_no_x_jax,
    mutation_kernel_jax,
    resampling_kernel_jax,
)
from ..utils.protein_nucleotide_utils import calculate_fitness_batch_jax

logger = getLogger(__name__)
logger.setLevel("INFO") # TODO: have this set from main script or config


def run_smc_protein_jax(
    prng_key_smc_steps: jax.Array, # PRNG key for the main SMC loop steps
    initial_population_key: jax.Array, # PRNG key for initial population mutation
    initial_population_mutation_rate: float, # Mutation kernel for initial population

    # --- Protein and Sequence Configuration ---
    protein_length: int,         # Integer, length of the protein in amino acids
    initial_sequence_char: str,    # String, initial sequence (AA or Nucleotide)
    sequence_type: str,          # String, 'protein' or 'nucleotide'
    mu_nuc: float,                 # Float, nucleotide mutation rate

    # --- Annealing Configuration ---
    annealing_schedule_func_py: Callable, # Python function for annealing schedule
    beta_max_val: float,           # Float, maximum beta value
    annealing_len_val: int,      # Integer, number of steps for annealing schedule

    # --- SMC Configuration ---
    n_particles: int,            # Integer, number of particles
    n_smc_steps: int,            # Integer, number of SMC steps

    # --- MPNN Model (passed directly) ---
    mpnn_model_instance: mk_mpnn_model,  # The prepped ColabDesign MPNN model object
    mpnn_model_is_active_static: bool,  # Boolean, if MPNN model is active
    
    save_traj: bool = False # Boolean, if True, save trajectory of particles
) -> dict[str, jax.Array]:
    """
    Runs a Sequential Monte Carlo simulation for protein sequence design,
    optimized for JAX.

    The core SMC loop is implemented using jax.lax.scan for JAX compatibility
    and performance. Initial sequence conversion and final metrics processing
    remain in Python. The function handles the generation of initial nucleotide
    sequences, mutation, fitness calculation, and resampling of particles, all
    while maintaining the constraints of the nucleotide alphabet. The function
    also includes a reporting mechanism for initial conditions and final results.
    
    Parameters:
    - prng_key_smc_steps: JAX PRNG key for the main SMC loop steps.
    - initial_population_key: JAX PRNG key for the initial population mutation.
    - protein_length: Length of the protein in amino acids.
    - initial_aa_seq_char: Initial amino acid sequence as a string.
    - mu_nuc: Nucleotide mutation rate.
    - annealing_schedule_func_py: Python function for the annealing schedule.
    - beta_max_val: Maximum beta value for the annealing schedule.
    - annealing_len_val: Number of steps for the annealing schedule to reach beta_max.
    - n_particles: Number of particles for the SMC simulation.
    - n_smc_steps: Number of SMC steps to perform.
    - mpnn_model_instance: Prepped ColabDesign MPNN model object.
    - mpnn_model_is_active_static: Boolean indicating if the MPNN model is active.
    - save_traj: Boolean indicating if the trajectory of particles should be saved.
    Returns:
    - final_results: Dictionary containing the results of the SMC simulation,
      including fitness metrics, entropy values, and other relevant statistics.
    """

    N_nuc_total = 3 * protein_length # Total number of nucleotides in a sequence
    schedule_name_str = annealing_schedule_func_py.__name__

    logger.info(
        f"Running Protein SMC (JAX) L={protein_length}, "
        f"Schedule={schedule_name_str}, BetaMax={beta_max_val}, "
        f"Particles={n_particles}, Steps={n_smc_steps}"
    )

    # 1. Initial Nucleotide Sequence TEMPLATE Preparation (Python part)
    # This template is X-free and matches the PDB sequence's preferred codons.
    if sequence_type == "protein":
        initial_nucleotide_seq_int_list = [NUCLEOTIDES_INT_MAP['A']] * N_nuc_total
        try:
            for i in range(protein_length):
                aa_char = initial_sequence_char[i]
                codon_char_list = list(RES_TO_CODON_CHAR[aa_char])
                for j in range(3):
                    initial_nucleotide_seq_int_list[3*i + j] = \
                        NUCLEOTIDES_INT_MAP[codon_char_list[j]]
        except KeyError as e:
            raise ValueError(
                f"Failed to generate initial JAX nucleotide template from AA '{e}'. "
                f"Check RES_TO_CODON_CHAR and initial_sequence_char."
            ) from e
        initial_template_one_seq_jax = jnp.array(
            initial_nucleotide_seq_int_list, dtype=jnp.int32
        )
    elif sequence_type == "nucleotide":
        initial_template_one_seq_jax = jnp.array(
            [NUCLEOTIDES_INT_MAP[nuc] for nuc in initial_sequence_char], dtype=jnp.int32
        )
    else:
        raise ValueError(f"Unsupported sequence_type: {sequence_type}")

    # Tile this X-free template for all particles
    _particles_jax_template_batch = jnp.tile(
        initial_template_one_seq_jax, (n_particles, 1)
    )

    # 2. Generate Initial Population using the NEW X-avoiding nucleotide mutation kernel
    # Apply a fixed initial mutation rate (e.g., 0.35 as in your original
    # script for this step)
    # This `mu_nuc` for initial mutation can be different from the `mu_nuc` for
    # SMC steps.

    particles_jax_initial = initial_mutation_kernel_no_x_jax(
        key=initial_population_key, # Dedicated key for initial population
        particles_template_batch=_particles_jax_template_batch,
        mu_nuc=initial_population_mutation_rate,
        n_nuc_alphabet_size=len(NUCLEOTIDES_CHAR),
        protein_length=protein_length
    )
    # `particles_jax_initial` will have random nucleotide mutations relative to
    # the template, potentially different amino acid sequences, but no 'X'
    # codons introduced at this stage.

    # --- START: ADDED CODE FOR INITIAL CONDITIONS REPORTING ---
    logger.info(
        "--- Reporting on True Initial Conditions (Post-Initial Mutation, "
        "Pre-SMC Loop) ---"
    )

    key_for_initial_metric_calc, _ = random.split(prng_key_smc_steps)
    logger.info(
        "Calculating initial conditions metrics... "
        "Split PRNG key for initial conditions metrics: %s",
        key_for_initial_metric_calc
    )

    (init_cond_fitness_values,
     init_cond_cai_values,
     init_cond_mpnn_values,
     init_cond_aa_seqs_int,
     init_cond_has_x_flags) = calculate_fitness_batch_jax(
                                    key_for_initial_metric_calc,
                                    particles_jax_initial,
                                    protein_length,
                                    sequence_type,
                                    mpnn_model_instance,
                                    mpnn_model_is_active_static
                                )

    init_cond_fitness_jnp = jnp.array(init_cond_fitness_values)
    init_cond_cai_jnp = jnp.array(init_cond_cai_values)
    init_cond_mpnn_jnp = jnp.array(init_cond_mpnn_values)
    init_cond_aa_seqs_jnp = jnp.array(init_cond_aa_seqs_int)
    init_cond_particles_jnp = jnp.array(particles_jax_initial)

    # Calculate metrics (uniformly weighted as no selection occurred)
    ic_valid_fitness_mask = jnp.isfinite(init_cond_fitness_jnp)
    ic_mean_fitness = jnp.mean(
        init_cond_fitness_jnp[ic_valid_fitness_mask]
    ) if jnp.any(ic_valid_fitness_mask) else jnp.nan
    ic_max_fitness = jnp.max(
        init_cond_fitness_jnp[ic_valid_fitness_mask]
    ) if jnp.any(ic_valid_fitness_mask) else jnp.nan

    ic_valid_cai_mask = (init_cond_cai_jnp > 0) & jnp.isfinite(init_cond_cai_jnp)
    ic_mean_cai = jnp.mean(
        init_cond_cai_jnp[ic_valid_cai_mask]
    ) if jnp.any(ic_valid_cai_mask) else jnp.nan

    ic_valid_mpnn_mask = jnp.isfinite(init_cond_mpnn_jnp)
    ic_mean_mpnn = jnp.mean(
        init_cond_mpnn_jnp[ic_valid_mpnn_mask]
    ) if jnp.any(ic_valid_mpnn_mask) else jnp.nan

    ic_nuc_entropy = calculate_nucleotide_entropy_py(init_cond_particles_jnp)
    ic_aa_entropy = calculate_amino_acid_entropy_py(init_cond_aa_seqs_jnp)
    ic_num_with_x = jnp.sum(jnp.array(init_cond_has_x_flags))

    print(f"  Mean Combined Fitness: {ic_mean_fitness:.4f}")
    print(f"  Max Combined Fitness: {ic_max_fitness:.4f}")
    print(f"  Mean CAI: {ic_mean_cai:.4f}")
    print(f"  Mean MPNN Score: {ic_mean_mpnn:.4f}")
    print(f"  Nucleotide Entropy: {ic_nuc_entropy:.4f}")
    print(f"  Amino Acid Entropy: {ic_aa_entropy:.4f}")
    print(f"  Number of sequences with X: {ic_num_with_x}")

    # --- END: ADDED CODE FOR INITIAL CONDITIONS REPORTING ---

    # 2. Precompute Beta Schedule (JAX array for scan)
    # `p_step_for_schedule` ranges from 1 to n_smc_steps
    beta_schedule_jax = jnp.array([
        annealing_schedule_func_py(p_step + 1, annealing_len_val, beta_max_val)
        for p_step in range(n_smc_steps)
    ], dtype=jnp.float32)

    # 3. Define the JAX-jitted SMC step function for jax.lax.scan
    # CORRECTED: The static arguments are now part of the function signature.
    @partial(jit, static_argnames=(
        'protein_length_static', 'n_particles_static', 'mu_nuc_static',
        'mpnn_model_instance_static', 'mpnn_model_is_active_static_flag',
        'N_nuc_alphabet_size_static', 'sequence_type_static'
    ))
    def _smc_scan_step(
        carry_state, scan_ijnputs_current_step, # Dynamic arguments from lax.scan
        # Static arguments (passed via lambda in lax.scan call)
        protein_length_static,
        n_particles_static,
        mu_nuc_static,
        mpnn_model_instance_static,
        mpnn_model_is_active_static_flag,
        N_nuc_alphabet_size_static,
        sequence_type_static
    ):
        """
        Performs one step of the SMC algorithm. Designed for jax.lax.scan.
        Static arguments are passed from the outer scope via the lambda in the
        scan call.
        """
        particles_prev_step, prev_log_Z_estimate, key_carry_prev = carry_state
        beta_current, _ = scan_ijnputs_current_step

        key_current_step_ops, key_for_next_carry = random.split(key_carry_prev)
        key_mutate_loop, key_fitness_loop, key_resample_loop = random.split(
            key_current_step_ops, 3
        )

        particles_mutated = mutation_kernel_jax(
            key_mutate_loop,
            particles_prev_step,
            mu_nuc_static,
            N_nuc=N_nuc_alphabet_size_static
        )

        fitness_values, cai_values, mpnn_values, aa_seqs_int, has_x_flags = \
            calculate_fitness_batch_jax(
                key_fitness_loop,
                particles_mutated,
                protein_length_static,
                sequence_type_static,
                mpnn_model_instance_static,
                mpnn_model_is_active_static_flag
            )

        log_weights = jnp.where(
            jnp.isneginf(fitness_values), -jnp.inf, beta_current * fitness_values
        )

        finite_log_weights_mask = jnp.isfinite(log_weights)
        num_finite_weights = jnp.sum(finite_log_weights_mask)

        mean_log_weights_for_Z_current_step = jnp.where(
            num_finite_weights > 0,
            jax.scipy.special.logsumexp(
                jnp.where(finite_log_weights_mask, log_weights, -jnp.inf)
            ) - jnp.log(num_finite_weights),
            jnp.where(beta_current == 0.0, 0.0, -jnp.inf)
        )
        current_log_Z_estimate = (
            prev_log_Z_estimate + mean_log_weights_for_Z_current_step
        )

        particles_resampled, ess_current_step, normalized_weights = \
            resampling_kernel_jax(
                key_resample_loop,
                particles_mutated,
                log_weights,
                n_particles_static
            )

        valid_fitness_mask_for_metrics = jnp.isfinite(fitness_values)
        sum_valid_weights = jnp.sum(
            jnp.where(valid_fitness_mask_for_metrics, normalized_weights, 0.0)
        )

        def safe_weighted_mean(
            metric_array, weights_array, valid_mask, sum_valid_w
        ):
            return jnp.where(
                sum_valid_w > 1e-9,
                jnp.sum(
                    jnp.where(valid_mask, metric_array * weights_array, 0.0)
                ) / sum_valid_w,
                jnp.nan
            )

        mean_combined_fitness = safe_weighted_mean(
            fitness_values, normalized_weights,
            valid_fitness_mask_for_metrics, sum_valid_weights
        )
        max_val = jnp.max(
            jnp.where(valid_fitness_mask_for_metrics, fitness_values, -jnp.inf)
        )
        max_combined_fitness = jnp.where(
            jnp.all(~valid_fitness_mask_for_metrics), jnp.nan, max_val
        )

        valid_cai_mask_for_metrics = (cai_values > 0) & jnp.isfinite(cai_values)
        sum_valid_cai_weights = jnp.sum(
            jnp.where(valid_cai_mask_for_metrics, normalized_weights, 0.0)
        )
        mean_cai = safe_weighted_mean(
            cai_values, normalized_weights, valid_cai_mask_for_metrics,
            sum_valid_cai_weights
        )

        valid_mpnn_mask_for_metrics = jnp.isfinite(mpnn_values)
        sum_valid_mpnn_weights = jnp.sum(
            jnp.where(valid_mpnn_mask_for_metrics, normalized_weights, 0.0)
        )
        mean_mpnn_score = safe_weighted_mean(
            mpnn_values, normalized_weights, valid_mpnn_mask_for_metrics,
            sum_valid_mpnn_weights
        )

        collected_metrics = {
            "mean_combined_fitness": mean_combined_fitness,
            "max_combined_fitness": max_combined_fitness,
            "mean_cai": mean_cai,
            "mean_mpnn_score": mean_mpnn_score,
            "ess": ess_current_step,
            "beta": beta_current,
            "particles_for_entropy": particles_mutated,
            "aa_seqs_for_entropy": aa_seqs_int,
            "final_log_Z_increment": mean_log_weights_for_Z_current_step
        }

        next_carry_state = (
            particles_resampled, current_log_Z_estimate, key_for_next_carry
        )

        return next_carry_state, collected_metrics

    # 4. Run the SMC loop using jax.lax.scan
    initial_carry = (
        particles_jax_initial,
        jnp.array(0.0, dtype=jnp.float32),
        prng_key_smc_steps
    )

    scan_over_ijnputs = (beta_schedule_jax, jnp.arange(n_smc_steps))

    # The lambda function correctly passes static arguments to _smc_scan_step.
    # The _smc_scan_step function's JIT decorator now correctly refers to its
    # own arguments.
    final_carry_state, collected_outputs_scan = lax.scan(
        lambda carry, scan_in: _smc_scan_step(
            carry, scan_in, # Dynamic args
            protein_length, # Static args from here
            n_particles,
            mu_nuc,
            mpnn_model_instance,
            mpnn_model_is_active_static,
            len(NUCLEOTIDES_CHAR),
            sequence_type
        ),
        initial_carry,
        scan_over_ijnputs,
        length=n_smc_steps
    )

    # Ujnpack results from scan
    final_particles_state, final_log_Z_estimate_jax, _ = final_carry_state

    # Convert collected JAX arrays to NumPy for Python-based processing and reporting
    mean_combined_fitness_per_gen_jnp = jnp.array(
        collected_outputs_scan["mean_combined_fitness"]
    )
    max_combined_fitness_per_gen_jnp = jnp.array(
        collected_outputs_scan["max_combined_fitness"]
    )
    mean_cai_per_gen_jnp = jnp.array(collected_outputs_scan["mean_cai"])
    mean_mpnn_score_per_gen_jnp = jnp.array(
        collected_outputs_scan["mean_mpnn_score"]
    )
    ess_per_gen_jnp = jnp.array(collected_outputs_scan["ess"])
    beta_per_gen_jnp = jnp.array(collected_outputs_scan["beta"])

    # Calculate entropy per generation using Python functions on the collected
    # particle states
    entropy_nuc_per_gen_jnp = jnp.full((n_smc_steps,), jnp.nan)
    entropy_aa_per_gen_jnp = jnp.full((n_smc_steps,), jnp.nan)

    for p_step in range(n_smc_steps):
        # Particles and AA sequences for entropy for this generation
        current_particles_for_entropy_jnp = jnp.array(
            collected_outputs_scan["particles_for_entropy"][p_step]
        )
        current_aa_seqs_for_entropy_jnp = jnp.array(
            collected_outputs_scan["aa_seqs_for_entropy"][p_step]
        )

        entropy_nuc_per_gen_jnp = calculate_nucleotide_entropy_py(
            current_particles_for_entropy_jnp
        )
        entropy_aa_per_gen_jnp = calculate_amino_acid_entropy_py(
            current_aa_seqs_for_entropy_jnp
        )

        # Print progress (optional, similar to original)
        if ((p_step + 1) % max(1, n_smc_steps // 10) == 0) or (p_step == 0):
            print(
                f"  Step {p_step+1}/{n_smc_steps} (JAX scan output): "
                f"MeanFit={mean_combined_fitness_per_gen_jnp[p_step]:.4f}, "
                f"MaxFit={max_combined_fitness_per_gen_jnp[p_step]:.4f}, "
                f"ESS={ess_per_gen_jnp[p_step]:.2f}"
            )

    # Final logZ estimate (sum of increments)
    log_Z_estimate_final_py = float(final_log_Z_estimate_jax)

    # Final amino acid entropy (based on the AA sequences of the very last
    # generation collected)
    if n_smc_steps > 0:
        final_aa_seqs_for_entropy_jnp = jnp.array(
            collected_outputs_scan["aa_seqs_for_entropy"][-1]
        )
        final_aa_entropy = calculate_amino_acid_entropy_py(
            final_aa_seqs_for_entropy_jnp
        )
    else:
        final_aa_entropy = jnp.nan

    # 5. Package Results
    final_results = {
        "protein_length": protein_length, "nucleotide_length": N_nuc_total,
        "initial_sequence": initial_sequence_char,
        "sequence_type": sequence_type,
        "mu_nuc": mu_nuc, "n_particles": n_particles, "n_smc_steps": n_smc_steps,
        "annealing_schedule": schedule_name_str, "annealing_len": annealing_len_val,
        "beta_max": beta_max_val,
        "final_logZhat": log_Z_estimate_final_py,
        "mean_combined_fitness_per_gen": mean_combined_fitness_per_gen_jnp,
        "max_combined_fitness_per_gen": max_combined_fitness_per_gen_jnp,
        "mean_cai_per_gen": mean_cai_per_gen_jnp,
        "mean_mpnn_score_per_gen": mean_mpnn_score_per_gen_jnp,
        "entropy_per_gen": entropy_nuc_per_gen_jnp, # Nucleotide entropy
        "aa_entropy_per_gen": entropy_aa_per_gen_jnp, # Amino acid entropy
        "beta_per_gen": beta_per_gen_jnp,
        "ess_per_gen": ess_per_gen_jnp,
        "final_amino_acid_entropy": final_aa_entropy,
        # Placeholders for metrics not fully implemented in this JAX version
        "adaptive_rate_per_gen": jnp.full((n_smc_steps,), jnp.nan),
        "final_var_V_hat_centred": jnp.nan,
        "final_var_v_hat_centred": jnp.nan,
        "jeffreys_divergence_nucleotide": jnp.nan,
        "jeffreys_divergence_amino_acid": jnp.nan,
    }

    print(
        f"Finished JAX SMC. Final MeanFit={mean_combined_fitness_per_gen_jnp[-1]:.4f}, "
        f"LogZhat={log_Z_estimate_final_py:.4f}"
    )

    return final_results