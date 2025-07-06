from dataclasses import dataclass
from functools import partial
from logging import getLogger
from typing import Literal

import jax
import jax.nn
import jax.numpy as jnp
from jax import jit, lax, random
from jaxtyping import PRNGKeyArray

from ..utils import (
  NUCLEOTIDES_CHAR,
  NUCLEOTIDES_INT_MAP,
  RES_TO_CODON_CHAR,
  AnnealingScheduleConfig,
  FitnessEvaluator,
  calculate_logZ_increment,
  calculate_population_fitness,
  diversify_initial_sequences,
  resample,
  shannon_entropy,
)
from ..utils.mutation import dispatch_mutation

logger = getLogger(__name__)
logger.setLevel("INFO")  # TODO: have this set from main script or config

# --- JAX-registered dataclasses for configuration and output ---


@dataclass(frozen=True)
class SMCConfig:
  sequence_length: int
  population_size: int
  mutation_rate: float
  sequence_type: Literal["protein", "nucleotide"]
  fitness_evaluator: FitnessEvaluator
  generations: int = 1000

  def tree_flatten(self):
    children = ()
    aux_data = self.__dict__
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(**aux_data)


@dataclass
class SMCOutput:
  mean_combined_fitness_per_gen: jnp.ndarray
  max_combined_fitness_per_gen: jnp.ndarray
  mean_cai_per_gen: jnp.ndarray
  mean_mpnn_score_per_gen: jnp.ndarray
  entropy_per_gen: jnp.ndarray
  aa_entropy_per_gen: jnp.ndarray
  beta_per_gen: jnp.ndarray
  ess_per_gen: jnp.ndarray
  final_logZhat: float
  final_amino_acid_entropy: float
  # Add more fields as needed

  def tree_flatten(self):
    children = (
      self.mean_combined_fitness_per_gen,
      self.max_combined_fitness_per_gen,
      self.mean_cai_per_gen,
      self.mean_mpnn_score_per_gen,
      self.entropy_per_gen,
      self.aa_entropy_per_gen,
      self.beta_per_gen,
      self.ess_per_gen,
    )
    aux_data = {
      "final_logZhat": self.final_logZhat,
      "final_amino_acid_entropy": self.final_amino_acid_entropy,
    }
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(
      mean_combined_fitness_per_gen=children[0],
      max_combined_fitness_per_gen=children[1],
      mean_cai_per_gen=children[2],
      mean_mpnn_score_per_gen=children[3],
      entropy_per_gen=children[4],
      aa_entropy_per_gen=children[5],
      beta_per_gen=children[6],
      ess_per_gen=children[7],
      **aux_data,
    )


@dataclass
class SMCCarryState:
  population: jax.Array
  logZ_estimate: jax.Array
  prng_key: PRNGKeyArray

  def tree_flatten(self):
    children = (self.population, self.logZ_estimate, self.prng_key)
    aux_data = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children)


@dataclass
class SMCScanInput:
  beta: jax.Array
  step_idx: jax.Array

  def tree_flatten(self):
    children = (self.beta, self.step_idx)
    aux_data = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children)


jax.tree_util.register_pytree_node_class(SMCConfig)
jax.tree_util.register_pytree_node_class(SMCOutput)
jax.tree_util.register_pytree_node_class(SMCCarryState)
jax.tree_util.register_pytree_node_class(SMCScanInput)


def smc_sampler(
  smc_config: SMCConfig,
  annealing_schedule_config: AnnealingScheduleConfig,
  prng_key_smc_steps: PRNGKeyArray,
  initial_population_key: PRNGKeyArray,
  diversification_ratio: float,
  initial_sequence: str,
  save_traj: bool = False,
) -> SMCOutput:
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
  - smc_config: SMCConfig dataclass with SMC parameters.
  - annealing_schedule_config: AnnealingScheduleConfig dataclass.
  - prng_key_smc_steps: JAX PRNG key for the main SMC loop steps.
  - initial_population_key: JAX PRNG key for the initial population mutation.
  - diversification_ratio: Mutation rate for initial population diversification.
  - initial_sequence: Initial sequence as a string.
  - save_traj: Boolean indicating if the trajectory of population should be saved.

  Returns:
  - final_results: Dictionary containing the results of the SMC simulation,
    including fitness metrics, entropy values, and other relevant statistics.
  """
  schedule_name_str = annealing_schedule_config.schedule_fn.__name__
  beta_max_val = annealing_schedule_config.beta_max
  annealing_len_val = annealing_schedule_config.annealing_len
  schedule_args = annealing_schedule_config.schedule_args
  population_size = smc_config.population_size
  generations = smc_config.generations

  logger.info(
    f"Running Protein SMC (JAX) L={len(initial_sequence)}, "
    f"Schedule={schedule_name_str}, ScheduleArgs={schedule_args}, "
    f"PopulationSize={population_size}, Steps={generations}"
  )

  if smc_config.sequence_type == "protein":
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
      if i < len(initial_sequence):
        aa_char = initial_sequence[i]
      else:
        aa_char = "unknown"
      error_message = (
        f"Failed to generate initial JAX nucleotide template from AA '{aa_char}'. "
        f"Check RES_TO_CODON_CHAR and initial_sequence."
      )
      raise ValueError(error_message) from e
    initial_single_sequence = jnp.array(initial_nucleotide_seq_int_list, dtype=jnp.int8)
  elif smc_config.sequence_type == "nucleotide":
    initial_single_sequence = jnp.array(
      [NUCLEOTIDES_INT_MAP[nuc] for nuc in initial_sequence], dtype=jnp.int8
    )
  else:
    raise ValueError(f"Unsupported sequence_type: {smc_config.sequence_type}")

  _template_population = jnp.tile(initial_single_sequence, (population_size, 1))

  # Create initial population
  initial_population = diversify_initial_sequences(
    key=initial_population_key,
    template_sequences=_template_population,
    mutation_rate=diversification_ratio,
    n_states=len(NUCLEOTIDES_CHAR),
    nucleotide=smc_config.sequence_type == "nucleotide",
  )

  logger.info("--- Reporting on True Initial Conditions (Post-Initial Mutation, Pre-SMC Loop) ---")

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
    smc_config.sequence_type,
    smc_config.fitness_evaluator,
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

  logger.info(f"  Mean Combined Fitness: {ic_mean_fitness:.4f}")
  logger.info(f"  Max Combined Fitness: {ic_max_fitness:.4f}")
  logger.info(f"  Mean CAI: {ic_mean_cai:.4f}")
  logger.info(f"  Mean MPNN Score: {ic_mean_mpnn:.4f}")
  logger.info(f"  Nucleotide Entropy: {ic_nuc_entropy:.4f}")

  # Create annealing schedule
  beta_schedule_jax = jnp.array(
    [
      annealing_schedule_config.schedule_fn(p_step + 1, annealing_len_val, beta_max_val)
      for p_step in range(generations)
    ],
    dtype=jnp.float32,
  )

  # Prepare initial carry state and scan inputs using dataclasses
  initial_carry = SMCCarryState(
    population=initial_population,
    logZ_estimate=jnp.array(0.0, dtype=jnp.float32),
    prng_key=prng_key_smc_steps,
  )
  scan_over_inputs = SMCScanInput(
    beta=beta_schedule_jax, step_idx=jnp.arange(generations, dtype=jnp.int32)
  )

  @partial(
    jit,
    static_argnames=("smc_config",),
  )
  def _smc_scan_step(
    carry_state: SMCCarryState,
    scan_input: SMCScanInput,
    smc_config: SMCConfig,
  ) -> tuple[SMCCarryState, dict[str, jax.Array]]:
    """
    Performs one step of the SMC algorithm. Designed for jax.lax.scan.
    """
    population_prev_step = carry_state.population
    prev_log_Z_estimate = carry_state.logZ_estimate
    key_carry_prev = carry_state.prng_key
    beta_current = scan_input.beta

    key_current_step_ops, key_for_next_carry = random.split(key_carry_prev)
    key_mutate_loop, key_fitness_loop, key_resample_loop = random.split(key_current_step_ops, 3)

    population_mutated = dispatch_mutation(
      key_mutate_loop,
      population_prev_step,
      smc_config.mutation_rate,
      smc_config.sequence_type,
    )

    fitness_values, fitness_components = calculate_population_fitness(
      key_fitness_loop,
      population_mutated,
      smc_config.sequence_type,
      smc_config.fitness_evaluator,
    )

    cai_values = fitness_components.get("cai", jnp.zeros_like(fitness_values))
    mpnn_values = fitness_components.get("mpnn", jnp.zeros_like(fitness_values))

    log_weights = jnp.where(jnp.isneginf(fitness_values), -jnp.inf, beta_current * fitness_values)

    if not isinstance(log_weights, jax.Array):
      error_message = (
        f"Expected log_weights to be a JAX array, got {type(log_weights)}. "
        "Ensure fitness_values is a JAX array."
      )
      logger.error(error_message)
      raise TypeError(error_message)

    mean_log_weights_for_Z_current_step = calculate_logZ_increment(log_weights, population_size)
    current_log_Z_estimate = prev_log_Z_estimate + mean_log_weights_for_Z_current_step

    population_resampled, ess_current_step, normalized_weights = resample(
      key_resample_loop, population_mutated, log_weights
    )

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
    if not isinstance(valid_fitness_mask_for_metrics, jax.Array):
      error_message = (
        f"Expected valid_fitness_mask_for_metrics to be a JAX array, got {type(valid_fitness_mask_for_metrics)}. "
        "Ensure fitness_values is a JAX array."
      )
      logger.error(error_message)
      raise TypeError(error_message)
    if not isinstance(fitness_values, jax.Array):
      error_message = (
        f"Expected mean_combined_fitness to be a JAX array, got {type(mean_combined_fitness)}. "
        "Ensure fitness_values and normalized_weights are JAX arrays."
      )
      logger.error(error_message)
      raise TypeError(error_message)
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

    next_carry_state = SMCCarryState(
      population=population_resampled,
      logZ_estimate=current_log_Z_estimate,
      prng_key=key_for_next_carry,
    )

    return next_carry_state, collected_metrics

  final_carry_state, collected_outputs_scan = lax.scan(
    lambda carry, scan_in: _smc_scan_step(carry, scan_in, smc_config),
    initial_carry,
    scan_over_inputs,
    length=generations,
  )

  final_population_state = final_carry_state.population
  final_log_Z_estimate_jax = final_carry_state.logZ_estimate

  if generations > 0:
    mean_combined_fitness_per_gen_jnp = jnp.array(collected_outputs_scan["mean_combined_fitness"])
    max_combined_fitness_per_gen_jnp = jnp.array(collected_outputs_scan["max_combined_fitness"])
    mean_cai_per_gen_jnp = jnp.array(collected_outputs_scan["mean_cai"])
    mean_mpnn_score_per_gen_jnp = jnp.array(collected_outputs_scan["mean_mpnn_score"])
    ess_per_gen_jnp = jnp.array(collected_outputs_scan["ess"])
    beta_per_gen_jnp = jnp.array(collected_outputs_scan["beta"])

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
      logger.info(
        f"  Step {p_step+1}/{generations} (JAX scan output): "
        f"MeanFit={mean_combined_fitness_per_gen_jnp[p_step]:.4f}, "
        f"MaxFit={max_combined_fitness_per_gen_jnp[p_step]:.4f}, "
        f"ESS={ess_per_gen_jnp[p_step]:.2f}"
      )

  # Final logZ estimate
  log_Z_estimate_final_py = float(final_log_Z_estimate_jax)

  # Final entropy calculations
  final_aa_entropy = shannon_entropy(final_population_state) if generations > 0 else jnp.nan

  # Package Results using SMCOutput dataclass
  output = SMCOutput(
    mean_combined_fitness_per_gen=mean_combined_fitness_per_gen_jnp,
    max_combined_fitness_per_gen=max_combined_fitness_per_gen_jnp,
    mean_cai_per_gen=mean_cai_per_gen_jnp,
    mean_mpnn_score_per_gen=mean_mpnn_score_per_gen_jnp,
    entropy_per_gen=entropy_nuc_per_gen_jnp,
    aa_entropy_per_gen=entropy_aa_per_gen_jnp,
    beta_per_gen=beta_per_gen_jnp,
    ess_per_gen=ess_per_gen_jnp,
    final_logZhat=log_Z_estimate_final_py,
    final_amino_acid_entropy=final_aa_entropy,
  )

  if generations > 0:
    logger.info(
      f"Finished JAX SMC. Final MeanFit={output.mean_combined_fitness_per_gen[-1]:.4f}, "
      f"LogZhat={output.final_logZhat:.4f}"
    )
  else:
    logger.info("Finished JAX SMC. No steps performed.")

  return output
