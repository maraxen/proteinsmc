from functools import partial
from logging import getLogger
from typing import Callable, Literal

import jax
import jax.nn
import jax.numpy as jnp
import jax.scipy.special as jsp
from jax import jit, lax, random

from ..utils import (
  NUCLEOTIDES_CHAR,
  NUCLEOTIDES_INT_MAP,
  RES_TO_CODON_CHAR,
  FitnessEvaluator,
  calculate_population_fitness,
  diversify_initial_sequences,
  resample,
  shannon_entropy,
)
from ..utils.mutate import dispatch_mutation

logger = getLogger(__name__)
logger.setLevel("INFO")  # TODO: have this set from main script or config


def run_smc_protein_jax(
  prng_key_smc_steps: jax.Array,
  initial_population_key: jax.Array,
  diversification_ratio: float,
  initial_sequence: str,
  sequence_type: Literal["protein", "nucleotide"],
  evolve_as: Literal["nucleotide", "protein"],
  mutation_rate: float,
  annealing_schedule_function: Callable,
  annealing_schedule_args: tuple,
  population_size: int,
  generations: int,
  fitness_evaluator: FitnessEvaluator,
  save_traj: bool = False,
) -> dict[str, jax.Array]:
  """
  Runs a Sequential Monte Carlo simulation for protein sequence design,
  optimized for JAX.

  The core SMC loop is implemented using jax.lax.scan for JAX compatibility
  and performance. Initial sequence conversion and final metrics processing
  remain in Python. The function handles the generation of initial nucleotide
  sequences, mutation, fitness calculation, and resampling of population, all
  while maintaining the constraints of the nucleotide alphabet. The function
  also includes a reporting mechanism for initial conditions and final results.

  Parameters:
  - prng_key_smc_steps: JAX PRNG key for the main SMC loop steps.
  - initial_population_key: JAX PRNG key for the initial population mutation.
  - diversification_ratio: Mutation rate for initial population diversification.
  - initial_sequence: Initial sequence as a string.
  - sequence_type: Type of initial sequence ("protein" or "nucleotide").
  - evolve_as: Type of evolution ("nucleotide" or "protein").
  - mutation_rate: Mutation rate during evolution.
  - annealing_schedule_function: Function for the annealing schedule.
  - annealing_schedule_args: Arguments for the annealing schedule.
  - population_size: Number of sequences for the SMC simulation.
  - generations: Number of SMC steps to perform.
  - fitness_evaluator: FitnessEvaluator containing fitness functions.
  - save_traj: Boolean indicating if the trajectory of population should be saved.

  Returns:
  - final_results: Dictionary containing the results of the SMC simulation,
    including fitness metrics, entropy values, and other relevant statistics.
  """
  schedule_name_str = annealing_schedule_function.__name__
  beta_max_val, annealing_len_val = annealing_schedule_args[:2]

  logger.info(
    f"Running Protein SMC (JAX) L={len(initial_sequence)}, "
    f"Schedule={schedule_name_str}, ScheduleArgs={annealing_schedule_args}, "
    f"PopulationSize={population_size}, Steps={generations}"
  )

  # Initialize sequences based on sequence type
  if sequence_type == "protein":
    initial_nucleotide_seq_int_list = [NUCLEOTIDES_INT_MAP["A"]] * (3 * len(initial_sequence))
    i = 0  # Initialize i before the loop
    try:
      for i in range(len(initial_sequence)):
        aa_char = initial_sequence[i]
        if aa_char not in RES_TO_CODON_CHAR or not RES_TO_CODON_CHAR[aa_char]:
          raise ValueError(
            f"No codons found for amino acid '{aa_char}'. "
            f"Check RES_TO_CODON_CHAR and initial_sequence."
          )
        codon_char_list = list(RES_TO_CODON_CHAR[aa_char][0])
        for j in range(3):
          initial_nucleotide_seq_int_list[3 * i + j] = NUCLEOTIDES_INT_MAP[codon_char_list[j]]
    except KeyError as e:
      # Use the last valid i from the loop, or handle empty sequence case
      if i < len(initial_sequence):
        aa_char = initial_sequence[i]
      else:
        aa_char = "unknown"
      error_message = (
        f"Failed to generate initial JAX nucleotide template from AA '{aa_char}'. "
        f"Check RES_TO_CODON_CHAR and initial_sequence."
      )
      raise ValueError(error_message) from e
    initial_template_one_seq_jax = jnp.array(initial_nucleotide_seq_int_list, dtype=jnp.int32)
  elif sequence_type == "nucleotide":
    initial_template_one_seq_jax = jnp.array(
      [NUCLEOTIDES_INT_MAP[nuc] for nuc in initial_sequence], dtype=jnp.int32
    )
  else:
    raise ValueError(f"Unsupported sequence_type: {sequence_type}")

  _template_population = jnp.tile(initial_template_one_seq_jax, (population_size, 1))

  # Create initial population
  initial_population = diversify_initial_sequences(
    key=initial_population_key,
    template_sequences=_template_population,
    mutation_rate=diversification_ratio,
    n_states=len(NUCLEOTIDES_CHAR),
    sequence_length=len(initial_sequence),
    nucleotide=(sequence_type == "nucleotide" or evolve_as == "nucleotide"),
  )

  logger.info(
    "--- Reporting on True Initial Conditions (Post-Initial Mutation, " "Pre-SMC Loop) ---"
  )

  key_for_initial_metric_calc, _ = random.split(prng_key_smc_steps)
  logger.info(
    "Calculating initial conditions metrics... "
    "Split PRNG key for initial conditions metrics: %s",
    key_for_initial_metric_calc,
  )

  # Calculate initial fitness using new system
  init_cond_fitness_values, init_cond_fitness_components = calculate_population_fitness(
    key_for_initial_metric_calc,
    initial_population,
    evolve_as,
    fitness_evaluator,
  )

  # Extract individual fitness components for reporting
  init_cond_cai_values = init_cond_fitness_components.get(
    "cai", jnp.zeros_like(init_cond_fitness_values)
  )
  init_cond_mpnn_values = init_cond_fitness_components.get(
    "mpnn", jnp.zeros_like(init_cond_fitness_values)
  )

  # Calculate initial metrics
  ic_mean_fitness = jnp.mean(init_cond_fitness_values)
  ic_max_fitness = jnp.max(init_cond_fitness_values)
  ic_mean_cai = jnp.mean(init_cond_cai_values)
  ic_mean_mpnn = jnp.mean(init_cond_mpnn_values)
  ic_nuc_entropy = shannon_entropy(initial_population)

  print(f"  Mean Combined Fitness: {ic_mean_fitness:.4f}")
  print(f"  Max Combined Fitness: {ic_max_fitness:.4f}")
  print(f"  Mean CAI: {ic_mean_cai:.4f}")
  print(f"  Mean MPNN Score: {ic_mean_mpnn:.4f}")
  print(f"  Nucleotide Entropy: {ic_nuc_entropy:.4f}")

  # Create annealing schedule
  beta_schedule_jax = jnp.array(
    [
      annealing_schedule_function(p_step + 1, annealing_len_val, beta_max_val)
      for p_step in range(generations)
    ],
    dtype=jnp.float32,
  )

  # Define the JAX-jitted SMC step function for jax.lax.scan
  @partial(
    jit,
    static_argnames=(
      "sequence_length_static",
      "population_size_static",
      "mutation_rate_static",
      "fitness_evaluator_static",
      "sequence_type_static",
      "evolve_as_static",
    ),
  )
  def _smc_scan_step(
    carry_state: tuple[jax.Array, jax.Array, jax.Array],
    scan_inputs_current_step: tuple[jax.Array, jax.Array],
    sequence_length_static: int,
    population_size_static: int,
    mutation_rate_static: float,
    fitness_evaluator_static: FitnessEvaluator,
    sequence_type_static: Literal["protein", "nucleotide"],
    evolve_as_static: Literal["nucleotide", "protein"],
  ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], dict[str, jax.Array]]:
    """
    Performs one step of the SMC algorithm. Designed for jax.lax.scan.
    """
    population_prev_step, prev_log_Z_estimate, key_carry_prev = carry_state
    beta_current, _ = scan_inputs_current_step

    key_current_step_ops, key_for_next_carry = random.split(key_carry_prev)
    key_mutate_loop, key_fitness_loop, key_resample_loop = random.split(key_current_step_ops, 3)

    # Apply mutations
    population_mutated = dispatch_mutation(
      key_mutate_loop,
      population_prev_step,
      mutation_rate_static,
      sequence_type_static,
      evolve_as_static,
    )

    # Calculate fitness using new system
    fitness_values, fitness_components = calculate_population_fitness(
      key_fitness_loop,
      population_mutated,
      evolve_as_static,
      fitness_evaluator_static,
    )

    # Extract individual components for reporting
    cai_values = fitness_components.get("cai", jnp.zeros_like(fitness_values))
    mpnn_values = fitness_components.get("mpnn", jnp.zeros_like(fitness_values))

    # Calculate weights and log evidence
    log_weights = jnp.where(jnp.isneginf(fitness_values), -jnp.inf, beta_current * fitness_values)

    if not isinstance(log_weights, jax.Array):
      error_message = (
        f"Expected log_weights to be a JAX array, got {type(log_weights)}. "
        "Ensure fitness_values is a JAX array."
      )
      logger.error(error_message)
      raise TypeError(error_message)

    finite_log_weights_mask = jnp.isfinite(log_weights)
    num_finite_weights = jnp.sum(finite_log_weights_mask)

    filtered_log_weights = jnp.where(finite_log_weights_mask, log_weights, -jnp.inf)
    mean_log_weights_for_Z_current_step = jnp.where(
      num_finite_weights > 0,
      jsp.logsumexp(filtered_log_weights) - jnp.log(num_finite_weights),
      jnp.where(beta_current == 0.0, 0.0, -jnp.inf),
    )
    current_log_Z_estimate = prev_log_Z_estimate + mean_log_weights_for_Z_current_step

    # Resample population
    population_resampled, ess_current_step, normalized_weights = resample(
      key_resample_loop, population_mutated, log_weights
    )

    # Calculate weighted metrics with isinstance checks
    valid_fitness_mask_for_metrics = jnp.isfinite(fitness_values)
    if isinstance(normalized_weights, jax.Array) and isinstance(
      valid_fitness_mask_for_metrics, jax.Array
    ):
      sum_valid_weights = jnp.sum(
        jnp.where(valid_fitness_mask_for_metrics, normalized_weights, 0.0)
      )
    else:
      sum_valid_weights = jnp.array(0.0)

    def safe_weighted_mean(
      metric_array: jax.Array,
      weights_array: jax.Array,
      valid_mask: jax.Array,
      sum_valid_w: jax.Array,
    ) -> jax.Array:
      return jnp.where(
        sum_valid_w > 1e-9,
        jnp.sum(jnp.where(valid_mask, metric_array * weights_array, 0.0)) / sum_valid_w,
        jnp.nan,
      )

    mean_combined_fitness = safe_weighted_mean(
      fitness_values, normalized_weights, valid_fitness_mask_for_metrics, sum_valid_weights
    )
    max_val = jnp.max(jnp.where(valid_fitness_mask_for_metrics, fitness_values, -jnp.inf))
    max_combined_fitness = jnp.where(jnp.all(~valid_fitness_mask_for_metrics), jnp.nan, max_val)

    valid_cai_mask_for_metrics = (cai_values > 0) & jnp.isfinite(cai_values)
    if isinstance(normalized_weights, jax.Array) and isinstance(
      valid_cai_mask_for_metrics, jax.Array
    ):
      sum_valid_cai_weights = jnp.sum(
        jnp.where(valid_cai_mask_for_metrics, normalized_weights, 0.0)
      )
    else:
      sum_valid_cai_weights = jnp.array(0.0)
    mean_cai = safe_weighted_mean(
      cai_values, normalized_weights, valid_cai_mask_for_metrics, sum_valid_cai_weights
    )

    valid_mpnn_mask_for_metrics = jnp.isfinite(mpnn_values)
    if isinstance(normalized_weights, jax.Array) and isinstance(
      valid_mpnn_mask_for_metrics, jax.Array
    ):
      sum_valid_mpnn_weights = jnp.sum(
        jnp.where(valid_mpnn_mask_for_metrics, normalized_weights, 0.0)
      )
    else:
      sum_valid_mpnn_weights = jnp.array(0.0)
    mean_mpnn_score = safe_weighted_mean(
      mpnn_values, normalized_weights, valid_mpnn_mask_for_metrics, sum_valid_mpnn_weights
    )

    collected_metrics = {
      "mean_combined_fitness": mean_combined_fitness,
      "max_combined_fitness": max_combined_fitness,
      "mean_cai": mean_cai,
      "mean_mpnn_score": mean_mpnn_score,
      "ess": ess_current_step,
      "beta": beta_current,
      "population_for_entropy": population_mutated,
      "final_log_Z_increment": mean_log_weights_for_Z_current_step,
    }

    next_carry_state = (population_resampled, current_log_Z_estimate, key_for_next_carry)

    return next_carry_state, collected_metrics

  # Run the SMC loop using jax.lax.scan
  initial_carry = (initial_population, jnp.array(0.0, dtype=jnp.float32), prng_key_smc_steps)
  scan_over_inputs = (beta_schedule_jax, jnp.arange(generations))

  final_carry_state, collected_outputs_scan = lax.scan(
    lambda carry, scan_in: _smc_scan_step(
      carry,
      scan_in,
      len(initial_sequence),
      population_size,
      mutation_rate,
      fitness_evaluator,
      sequence_type,
      evolve_as,
    ),
    initial_carry,
    scan_over_inputs,
    length=generations,
  )

  # Unpack results from scan
  final_population_state, final_log_Z_estimate_jax, _ = final_carry_state

  # Convert collected JAX arrays for reporting
  if generations > 0:
    mean_combined_fitness_per_gen_jnp = jnp.array(collected_outputs_scan["mean_combined_fitness"])
    max_combined_fitness_per_gen_jnp = jnp.array(collected_outputs_scan["max_combined_fitness"])
    mean_cai_per_gen_jnp = jnp.array(collected_outputs_scan["mean_cai"])
    mean_mpnn_score_per_gen_jnp = jnp.array(collected_outputs_scan["mean_mpnn_score"])
    ess_per_gen_jnp = jnp.array(collected_outputs_scan["ess"])
    beta_per_gen_jnp = jnp.array(collected_outputs_scan["beta"])

    # Calculate entropy per generation
    entropy_nuc_per_gen_jnp = jnp.array(
      [
        shannon_entropy(collected_outputs_scan["population_for_entropy"][p_step])
        for p_step in range(generations)
      ]
    )
    entropy_aa_per_gen_jnp = jnp.full((generations,), jnp.nan)  # Placeholder
  else:
    mean_combined_fitness_per_gen_jnp = jnp.array([])
    max_combined_fitness_per_gen_jnp = jnp.array([])
    mean_cai_per_gen_jnp = jnp.array([])
    mean_mpnn_score_per_gen_jnp = jnp.array([])
    ess_per_gen_jnp = jnp.array([])
    beta_per_gen_jnp = jnp.array([])
    entropy_nuc_per_gen_jnp = jnp.array([])
    entropy_aa_per_gen_jnp = jnp.array([])

  # Print progress
  for p_step in range(generations):
    if ((p_step + 1) % max(1, generations // 10) == 0) or (p_step == 0):
      print(
        f"  Step {p_step+1}/{generations} (JAX scan output): "
        f"MeanFit={mean_combined_fitness_per_gen_jnp[p_step]:.4f}, "
        f"MaxFit={max_combined_fitness_per_gen_jnp[p_step]:.4f}, "
        f"ESS={ess_per_gen_jnp[p_step]:.2f}"
      )

  # Final logZ estimate
  log_Z_estimate_final_py = float(final_log_Z_estimate_jax)

  # Final entropy calculations
  final_aa_entropy = shannon_entropy(final_population_state) if generations > 0 else jnp.nan

  # Package Results
  final_results = {
    "protein_length": len(initial_sequence),
    "nucleotide_length": len(initial_sequence) * 3,
    "initial_sequence": initial_sequence,
    "sequence_type": sequence_type,
    "mutation_rate": mutation_rate,
    "population_size": population_size,
    "generations": generations,
    "annealing_schedule": schedule_name_str,
    "annealing_len": annealing_len_val,
    "beta_max": beta_max_val,
    "final_logZhat": log_Z_estimate_final_py,
    "mean_combined_fitness_per_gen": mean_combined_fitness_per_gen_jnp,
    "max_combined_fitness_per_gen": max_combined_fitness_per_gen_jnp,
    "mean_cai_per_gen": mean_cai_per_gen_jnp,
    "mean_mpnn_score_per_gen": mean_mpnn_score_per_gen_jnp,
    "entropy_per_gen": entropy_nuc_per_gen_jnp,
    "aa_entropy_per_gen": entropy_aa_per_gen_jnp,
    "beta_per_gen": beta_per_gen_jnp,
    "ess_per_gen": ess_per_gen_jnp,
    "final_amino_acid_entropy": final_aa_entropy,
    # Placeholders for metrics not fully implemented in this JAX version
    "adaptive_rate_per_gen": jnp.full((generations,), jnp.nan),
    "final_var_V_hat_centred": jnp.nan,
    "final_var_v_hat_centred": jnp.nan,
    "jeffreys_divergence_nucleotide": jnp.nan,
    "jeffreys_divergence_amino_acid": jnp.nan,
  }

  if generations > 0:
    print(
      f"Finished JAX SMC. Final MeanFit={mean_combined_fitness_per_gen_jnp[-1]:.4f}, "
      f"LogZhat={log_Z_estimate_final_py:.4f}"
    )
  else:
    print("Finished JAX SMC. No steps performed.")

  return final_results
