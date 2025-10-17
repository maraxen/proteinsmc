"""Main entry point for running experiments."""

from __future__ import annotations

import logging
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import jax
import jax.numpy as jnp

from proteinsmc.io import RunManager
from proteinsmc.models import (
  GibbsConfig,
  HMCConfig,
  MCMCConfig,
  NUTSConfig,
  ParallelReplicaConfig,
  SMCConfig,
)
from proteinsmc.sampling import (
  run_gibbs_loop,
  run_hmc_loop,
  run_mcmc_loop,
  run_nuts_loop,
)
from proteinsmc.sampling.initialization_factory import initialize_sampler_state
from proteinsmc.sampling.particle_systems.parallel_replica import (
  run_prsmc_loop,
)
from proteinsmc.sampling.particle_systems.smc import run_smc_loop
from proteinsmc.utils.annealing import get_annealing_function
from proteinsmc.utils.constants import NUCLEOTIDES_NUM_STATES
from proteinsmc.utils.fitness import get_fitness_function
from proteinsmc.utils.memory import auto_tune_chunk_size
from proteinsmc.utils.mutation import mutate
from proteinsmc.utils.translation import aa_to_nucleotide, nucleotide_to_aa, string_to_int_sequence

if TYPE_CHECKING:
  from proteinsmc.models.sampler_base import BaseSamplerConfig
  from proteinsmc.models.translation import TranslateFuncSignature
  from proteinsmc.utils.fitness import StackedFitnessFn

from proteinsmc.models.sampler_base import config_to_jax

# The SAMPLER_REGISTRY maps a sampler type string to its configuration class
# and the core execution function. The initialization is now handled by a
# unified factory function.
SAMPLER_REGISTRY: dict[str, dict[str, Any]] = {
  "smc": {
    "config_cls": SMCConfig,
    "run_fn": run_smc_loop,
  },
  "parallel_replica": {
    "config_cls": ParallelReplicaConfig,
    "run_fn": run_prsmc_loop,
  },
  "gibbs": {
    "config_cls": GibbsConfig,
    "run_fn": run_gibbs_loop,
  },
  "mcmc": {
    "config_cls": MCMCConfig,
    "run_fn": run_mcmc_loop,
  },
  "hmc": {
    "config_cls": HMCConfig,
    "run_fn": run_hmc_loop,
  },
  "nuts": {
    "config_cls": NUTSConfig,
    "run_fn": run_nuts_loop,
  },
}


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _setup_fitness_function(
  key: jax.Array,
  config: BaseSamplerConfig,
) -> tuple[jax.Array, StackedFitnessFn]:
  """Configure the fitness function, applying auto-tuning for chunk size if enabled."""
  translate_func: TranslateFuncSignature = (
    nucleotide_to_aa if config.n_states == NUCLEOTIDES_NUM_STATES else aa_to_nucleotide
  )
  chunk_size = None
  fitness_fn = get_fitness_function(
    evaluator_config=config.fitness_evaluator,
    n_states=config.n_states,
    translate_func=translate_func,
  )

  if config.memory_config.auto_tuning_config.enable_auto_tuning:
    logger.info(
      "Auto-tuning chunk size for fitness function. This may take a moment.",
    )
    key, tune_key = jax.random.split(key)
    # Create dummy data for auto-tuning. If the run is batched, the fitness
    # function is expected to handle inputs with a leading batch dimension.
    num_rates = jnp.atleast_1d(jnp.array(config.mutation_rate)).shape[0]
    is_batched = num_rates > 1

    test_data_shape = (
      max(config.memory_config.auto_tuning_config.probe_chunk_sizes),
      len(config.seed_sequence),
    )
    if is_batched:
      test_data_shape = (num_rates, *test_data_shape)

    test_data = (
      tune_key,
      jnp.zeros(test_data_shape, dtype=jnp.int8),
    )
    chunk_size = auto_tune_chunk_size(
      func=fitness_fn,
      test_data=test_data,
      config=config.memory_config.auto_tuning_config,
    )
    logger.info("Auto-tuning selected chunk size: %d", chunk_size)

    # Re-create the fitness function with the optimal chunk size
    fitness_fn = get_fitness_function(
      evaluator_config=config.fitness_evaluator,
      n_states=config.n_states,
      translate_func=translate_func,
      chunk_size=chunk_size,
    )

  return key, fitness_fn


def _setup_mutation_function(config: BaseSamplerConfig) -> Callable:
  """Create a mutation function that handles mutation rates via update_parameters."""

  def mutation_fn(
    key: jax.Array,
    particles: jax.Array,
    update_parameters: dict,
  ) -> tuple[jax.Array, dict]:
    """Apply mutation rates from the state's update_parameters.

    Blackjax's `smc_step` automatically handles mapping this function over the
    batch dimension if the state is batched, passing a scalar `mutation_rate`
    from the `update_parameters` pytree for each parallel run.
    """
    mutated_particles = mutate(
      key=key,
      sequence=particles,
      mutation_rate=update_parameters["mutation_rate"],
      q_states=config.n_states,
    )
    return mutated_particles, {}

  return mutation_fn


def _validate_config(config: BaseSamplerConfig) -> dict[str, Any]:
  """Validate the sampler type and config object match."""
  if config.sampler_type not in SAMPLER_REGISTRY:
    msg = (
      f"Unknown sampler_type: '{config.sampler_type}'. "
      f"Available types: {list(SAMPLER_REGISTRY.keys())}."
    )
    raise ValueError(msg)

  sampler_def = SAMPLER_REGISTRY[config.sampler_type]
  if not isinstance(config, sampler_def["config_cls"]):
    msg = (
      f"Configuration object of type {type(config)} does not match "
      f"sampler type '{config.sampler_type}' which requires {sampler_def['config_cls']}."
    )
    raise TypeError(msg)

  return sampler_def


def _get_inputs(config: BaseSamplerConfig) -> tuple[dict[str, Any], list[str]]:
  """Prepare JAX-compatible inputs from the configuration."""
  jax_inputs = config_to_jax(config)

  def _convert_to_list(
    item: str | Sequence[str],
  ) -> list[str]:
    """Convert a string or sequence of strings to a list of strings."""
    if isinstance(item, str):
      return [item]
    return list(item)

  seed_sequence_inputs = _convert_to_list(config.seed_sequence)
  sequence_type_inputs = _convert_to_list(config.sequence_type)
  seed_sequences = jnp.array(
    [
      string_to_int_sequence(
        seq,
        None,
        stype,  # type: ignore[arg-type]
      )
      for seq, stype in (
        zip(
          seed_sequence_inputs,
          sequence_type_inputs,
        )
        if config.combinations_mode == "zip"
        else product(
          seed_sequence_inputs,
          sequence_type_inputs,
        )
      )
    ],
  )
  jax_inputs["seed_sequences"] = seed_sequences
  return jax_inputs, sequence_type_inputs


def run_experiment(config: BaseSamplerConfig, output_dir: str | Path, seed: int = 0) -> None:
  """Run a sampling experiment based on the provided configuration."""
  key = jax.random.PRNGKey(seed)

  # 1. Validate config and get sampler-specific functions
  sampler_def = _validate_config(config)
  run_fn: Callable[..., tuple[Any, dict[str, Any]]] = sampler_def["run_fn"]

  # 2. Set up fitness function and optional auto-tuning
  key, fitness_fn = _setup_fitness_function(key, config)

  # 3. Prepare other shared components
  mutation_fn = _setup_mutation_function(config)
  annealing_fn = (
    get_annealing_function(config.annealing_config)
    if hasattr(config, "annealing_config") and config.annealing_config is not None
    else None
  )

  with RunManager(Path(output_dir), config) as writer:
    logger.info(
      "Starting run %s of type '%s'...",
      writer.run_id,
      config.sampler_type,
    )
    jax_inputs, sequence_type_inputs = _get_inputs(config)
    key, init_key = jax.random.split(key)
    initial_states = (
      initialize_sampler_state(
        sampler_type=config.sampler_type,
        sequence_type=stype,  # type: ignore[arg-type]
        seed_sequence=jax_inputs["seed_sequences"],
        population_size=jax_inputs.get("population_size", None),
        algorithm=getattr(config, "algorithm", None),
        smc_algo_kwargs=getattr(config, "smc_algo_kwargs", {}),
        n_islands=jax_inputs.get("n_islands", None),
        population_size_per_island=jax_inputs.get("population_size_per_island", None),
        island_betas=getattr(config, "island_betas", None),
        diversification_ratio=jax_inputs.get("diversification_ratio", None),
        key=init_key,
        beta=jax_inputs.get("initial_beta", None),
        fitness_fn=fitness_fn,
        track_lineage=config.track_lineage,
      )
      for stype in sequence_type_inputs
    )

    # 5. Run the core sampler loop.
    logger.info("Starting sampler loop...")
    final_state, all_outputs = run_fn(
      config=config,
      initial_state=initial_states,
      fitness_fn=fitness_fn,
      mutation_fn=mutation_fn,
      annealing_fn=annealing_fn,
    )
    jax.block_until_ready(final_state)
    logger.info("Sampler loop finished.")

    # 6. Write results to disk
    logger.info("Writing results to disk...")
    # The output leaves now have a shape of (num_steps, batch_size, ...) if batched
    num_steps = jax.tree_util.tree_leaves(all_outputs)[0].shape[0]
    mutation_rates = jnp.atleast_1d(jnp.asarray(config.mutation_rate))
    is_batched = mutation_rates.shape[0] > 1

    for i in range(num_steps):
      step_output_tree = jax.tree_util.tree_map(lambda x: x[i], all_outputs)
      writer.step(step_output_tree)

      # Log scalar metrics for monitoring. If batched, log metrics from the first run.
      if is_batched:
        first_run_in_batch = jax.tree_util.tree_map(
          lambda x: x[0],
          step_output_tree,
        )
      else:
        first_run_in_batch = step_output_tree

      scalar_metrics: dict[str, int | float] = {"step": i}
      for metric_name, metric_value in first_run_in_batch.items():
        if jnp.ndim(metric_value) == 0:
          scalar_metrics[metric_name] = float(metric_value)
      writer.log_scalars(scalar_metrics)

  logger.info("Run %s completed successfully.", writer.run_id)
