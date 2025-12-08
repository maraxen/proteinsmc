"""OED experiment runner module.

This module contains helpers to run small, testable OED experiments on NK
landscapes. The implementation now reads from actual SMC outputs stored on disk
to calculate metrics from the full sampling trajectory.
"""

from pathlib import Path

import jax
import jax.numpy as jnp

from proteinsmc.io import read_lineage_data_range
from proteinsmc.models import (
  AutoTuningConfig,
  FitnessEvaluator,
  MemoryConfig,
  SMCConfig,
)
from proteinsmc.models.fitness import CombineFunction, FitnessFunction
from proteinsmc.oed.nk import create_landscape_from_design
from proteinsmc.oed.structs import OEDDesign, OEDPredictedVariables
from proteinsmc.oed.tracking import (
  get_run_records_range,
  get_shared_arrayrecord_path,
)
from proteinsmc.runner import run_experiment
from proteinsmc.utils import metrics
from proteinsmc.utils.serialization import create_sampler_output_skeleton

# Threshold for deciding whether an alphabet size indicates a protein
ALPHABET_THRESHOLD = 4


def convert_design_to_config(
  design: OEDDesign,
  fitness_evaluator: FitnessEvaluator,
  seed: int = 42,
  *,
  nucleotide_enabled: bool = False,
) -> SMCConfig:
  """Convert OED design parameters to sampler configuration.

  Args:
      design: OED design parameters.
      fitness_evaluator: Fitness evaluator for the experiment.
      seed: Random seed.
      nucleotide_enabled: Whether nucleotide sequences are enabled.

  Returns:
      SMCConfig object configured according to design parameters.

  """
  # Disable auto-tuning for OED experiments to avoid chunking issues
  auto_tuning_config = AutoTuningConfig(enable_auto_tuning=False)
  memory_config = MemoryConfig(auto_tuning_config=auto_tuning_config)
  sequence = jax.random.choice(
    jax.random.PRNGKey(seed), jnp.arange(design.q), shape=(design.N,), replace=True
  ).astype(jnp.int8)
  return SMCConfig(
    prng_seed=seed,
    seed_sequence=sequence,
    num_samples=design.n_generations,
    n_states=design.q,
    mutation_rate=design.mutation_rate,
    diversification_ratio=design.diversification_ratio,
    sequence_type="protein"
    if design.q > ALPHABET_THRESHOLD or not nucleotide_enabled
    else "nucleotide",
    fitness_evaluator=fitness_evaluator,
    population_size=design.population_size,
    memory_config=memory_config,
    annealing_config=None,
  )


def run_oed_experiment(
  design: OEDDesign, output_dir: str, seed: int = 42
) -> tuple[OEDPredictedVariables, str]:
  """Run OED experiment and calculate metrics.

  Args:
      design: OED design parameters.
      output_dir: Directory to store experiment results.
      seed: Random seed.

  Returns:
      Tuple of (OEDPredictedVariables with calculated metrics, run UUID).

  """
  # Ensure output directory exists (kept for API compatibility)
  Path(output_dir).mkdir(parents=True, exist_ok=True)

  # Create fitness evaluator (using NK landscape)
  key = jax.random.PRNGKey(seed)
  key, subkey = jax.random.split(key)
  landscape = create_landscape_from_design(subkey, design)
  # Build a FitnessEvaluator that uses the NK scoring function. This plugs
  # into the existing `get_fitness_function` machinery used by the runner.
  nk_fn = FitnessFunction(
    name="nk", n_states=design.q, kwargs={"landscape": landscape, "n": design.N, "k": design.K}
  )
  combine_fn = CombineFunction(name="sum")
  fitness_evaluator = FitnessEvaluator(fitness_functions=(nk_fn,), combine_fn=combine_fn)

  # Create a sampler config and call the main runner. The runner performs
  # SMC and writes outputs to disk in `output_dir`.
  config = convert_design_to_config(design, fitness_evaluator, seed)
  # run_experiment has side effects (I/O). It will create metadata and run the sampler.
  # It now returns the run UUID which we use to locate the output file.
  run_uuid = run_experiment(config, output_dir, seed, use_uuid=False)

  # Use tracking utilities to get the record range for this run from the shared ArrayRecord
  record_range = get_run_records_range(output_dir, run_uuid)
  if record_range is None:
    msg = f"Could not find record range for run UUID {run_uuid}"
    raise ValueError(msg)

  start_idx, end_idx = record_range

  # Read the actual SMC output from the shared ArrayRecord file
  shared_arrayrecord_path = get_shared_arrayrecord_path(output_dir)
  skeleton = create_sampler_output_skeleton(config)
  records = list(
    read_lineage_data_range(str(shared_arrayrecord_path), start_idx, end_idx, skeleton)
  )

  if not records:
    msg = f"No records found for run {run_uuid} in range [{start_idx}:{end_idx}]"
    raise ValueError(msg)

  # Extract first and last generation data
  # Records contain stacked generations: shape (n_generations, population_size, ...)
  first_record = records[0]
  last_record = records[-1] if len(records) > 1 else first_record

  # Get sequences and fitness from first and last steps
  # Shape: (n_generations, population_size, N) for sequences
  # Shape: (n_generations, population_size) for fitness
  initial_sequences = first_record["sequences"][0]  # First generation
  initial_fitness = first_record["fitness"][0]
  final_sequences = last_record["sequences"][-1]  # Last generation
  final_fitness = last_record["fitness"][-1]

  # Build fitness history across all records (mean fitness per generation)
  # fitness_history should be shape (n_generations,) for barrier crossing calculation
  fitness_history_full = jnp.concatenate([rec["fitness"] for rec in records], axis=0)
  fitness_history = jnp.mean(fitness_history_full, axis=1)  # Average across population

  # Calculate metrics using existing utilities
  # For information gain, use histogram-based approach on fitness distributions
  n_bins = 20
  initial_fitness_hist, _ = jnp.histogram(initial_fitness, bins=n_bins, density=False)
  final_fitness_hist, _ = jnp.histogram(final_fitness, bins=n_bins, density=False)
  info_gain = metrics.jeffreys_divergence(
    initial_fitness_hist.astype(jnp.float32),
    final_fitness_hist.astype(jnp.float32),
  )

  barrier_freq = metrics.calculate_barrier_crossing_frequency(fitness_history)
  final_entropy = metrics.shannon_entropy(final_sequences)

  # Compute JSD between flattened empirical counts of symbols
  p = jnp.bincount(initial_sequences.ravel(), minlength=design.q).astype(jnp.float32)
  q = jnp.bincount(final_sequences.ravel(), minlength=design.q).astype(jnp.float32)
  jsd = metrics.jensen_shannon_divergence(p, q)

  # Calculate geometric mean of fitness
  geometric_fitness_mean = metrics.calculate_geometric_fitness_mean(fitness_history)

  predicted_vars = OEDPredictedVariables(
    information_gain=info_gain,
    barrier_crossing_frequency=barrier_freq,
    final_sequence_entropy=final_entropy,
    jsd_from_original_population=jsd,
    geometric_fitness_mean=geometric_fitness_mean,
  )

  return predicted_vars, run_uuid
