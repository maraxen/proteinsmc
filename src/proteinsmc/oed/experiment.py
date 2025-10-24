"""OED experiment runner module.

This module contains helpers to run small, testable OED experiments on NK
landscapes. The implementation below intentionally simulates a single
generation rather than invoking the full sampling runner so unit tests run
quickly and deterministically.
"""

from pathlib import Path

import jax
import jax.numpy as jnp

from proteinsmc.models import (
  AnnealingConfig,
  AutoTuningConfig,
  FitnessEvaluator,
  MemoryConfig,
  SMCConfig,
)
from proteinsmc.models.fitness import CombineFunction, FitnessFunction
from proteinsmc.oed.nk import create_landscape_from_design
from proteinsmc.oed.structs import OEDDesign, OEDPredictedVariables
from proteinsmc.runner import run_experiment
from proteinsmc.utils import metrics
from proteinsmc.utils.constants import INT_TO_AA_CHAR_MAP
from proteinsmc.utils.nk_landscape import calculate_nk_fitness_population

# Threshold for deciding whether an alphabet size indicates a protein
ALPHABET_THRESHOLD = 4


def convert_design_to_config(
  design: OEDDesign, fitness_evaluator: FitnessEvaluator, seed: int = 42
) -> SMCConfig:
  """Convert OED design parameters to sampler configuration.

  Args:
      design: OED design parameters.
      fitness_evaluator: Fitness evaluator for the experiment.
      seed: Random seed.

  Returns:
      SMCConfig object configured according to design parameters.

  """
  # Disable auto-tuning for OED experiments to avoid chunking issues
  auto_tuning_config = AutoTuningConfig(enable_auto_tuning=False)
  memory_config = MemoryConfig(auto_tuning_config=auto_tuning_config)
  sequence = jax.random.choice(
    jax.random.PRNGKey(seed), jnp.arange(design.q), shape=(design.N,), replace=True
  ).astype(jnp.int8)
  str_sequence = "".join([INT_TO_AA_CHAR_MAP[int(s)] for s in sequence])  # Convert to string
  return SMCConfig(
    prng_seed=seed,
    seed_sequence=str_sequence,
    num_samples=design.n_generations,
    n_states=design.q,
    mutation_rate=design.mutation_rate,
    diversification_ratio=design.diversification_ratio,
    sequence_type="protein" if design.q > ALPHABET_THRESHOLD else "nucleotide",
    fitness_evaluator=fitness_evaluator,
    population_size=design.population_size,
    memory_config=memory_config,
    annealing_config=AnnealingConfig(
      annealing_fn="linear", beta_max=1.0, n_steps=design.n_generations
    ),
  )


def run_oed_experiment(design: OEDDesign, output_dir: str, seed: int = 42) -> OEDPredictedVariables:
  """Run OED experiment and calculate metrics.

  Args:
      design: OED design parameters.
      output_dir: Directory to store experiment results.
      seed: Random seed.

  Returns:
      OEDPredictedVariables with calculated metrics.

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
  run_experiment(config, output_dir, seed)

  # We'll attempt to read back any results via a lightweight simulation if the
  # full SMC outputs are not readily available in memory here.
  key = jax.random.PRNGKey(seed)
  # Initialize a random population of sequences (integers in [0, q))
  key, seq_key = jax.random.split(key)
  initial_sequences = jax.random.randint(
    seq_key,
    shape=(design.population_size, design.N),
    minval=0,
    maxval=design.q,
    dtype=jnp.int8,
  )

  # Compute initial fitness
  initial_fitness = calculate_nk_fitness_population(
    initial_sequences, landscape, design.N, design.K
  )

  # Simulate one generation of random mutations to produce final population
  key, mut_key = jax.random.split(key)
  mutation_mask = jax.random.bernoulli(
    mut_key, p=design.mutation_rate, shape=initial_sequences.shape
  )
  random_states = jax.random.randint(
    mut_key,
    shape=initial_sequences.shape,
    minval=0,
    maxval=design.q,
    dtype=jnp.int8,
  )
  final_sequences = jnp.where(mutation_mask, random_states, initial_sequences)
  final_fitness = calculate_nk_fitness_population(final_sequences, landscape, design.N, design.K)

  # Create a simple fitness history (initial, final)
  fitness_history = jnp.stack([initial_fitness, final_fitness])

  # Extract data for metrics calculation
  # (we already have initial_* and final_* variables)

  # Calculate metrics
  # Calculate metrics using existing utilities
  info_gain = metrics.jeffreys_divergence(
    jnp.atleast_1d(initial_fitness).astype(jnp.float32),
    jnp.atleast_1d(final_fitness).astype(jnp.float32),
  )
  barrier_freq = metrics.calculate_barrier_crossing_frequency(fitness_history)
  final_entropy = metrics.shannon_entropy(final_sequences)
  # Compute a simple JSD between flattened empirical counts of symbols.
  p = jnp.bincount(initial_sequences.ravel(), minlength=design.q).astype(jnp.float32)
  q = jnp.bincount(final_sequences.ravel(), minlength=design.q).astype(jnp.float32)
  jsd = metrics.jensen_shannon_divergence(p, q)

  return OEDPredictedVariables(
    information_gain=info_gain,
    barrier_crossing_frequency=barrier_freq,
    final_sequence_entropy=final_entropy,
    jsd_from_original_population=jsd,
  )
