"""Main entry point for running experiments."""

from __future__ import annotations

import logging
import uuid
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

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
from proteinsmc.utils.memory import auto_tune_batch_size
from proteinsmc.utils.mutation import mutate
from proteinsmc.utils.translation import aa_to_nucleotide, nucleotide_to_aa, string_to_int_sequence

if TYPE_CHECKING:
  from collections.abc import Callable, Sequence

  from array_record.python.array_record_module import ArrayRecordWriter

  from proteinsmc.models.sampler_base import BaseSamplerConfig
  from proteinsmc.models.translation import TranslateFuncSignature
  from proteinsmc.models.types import BatchEvoSequence, EvoSequence
  from proteinsmc.utils.fitness import StackedFitnessFn

from proteinsmc.io import create_metadata_file, create_writer_callback
from proteinsmc.models.sampler_base import config_to_jax

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


def _setup_writer_callback(path: Path) -> tuple[ArrayRecordWriter, Callable]:
  """Set up the writer callback for data tracking."""
  writer, writer_callback = create_writer_callback(str(path))

  return writer, writer_callback


def _setup_fitness_function(
  key: jax.Array,
  config: BaseSamplerConfig,
) -> tuple[jax.Array, StackedFitnessFn]:
  """Configure the fitness function, applying auto-tuning for chunk size if enabled."""
  translate_func: TranslateFuncSignature = (
    nucleotide_to_aa if config.n_states == NUCLEOTIDES_NUM_STATES else aa_to_nucleotide
  )
  batch_size = None
  fitness_fn = get_fitness_function(
    evaluator_config=config.fitness_evaluator,
    n_states=config.n_states,
    translate_func=translate_func,
    batch_size=batch_size,
  )

  if config.memory_config.auto_tuning_config.enable_auto_tuning:
    logger.info(
      "Auto-tuning chunk size for fitness function. This may take a moment.",
    )
    key, tune_key = jax.random.split(key)
    num_rates = jnp.atleast_1d(jnp.array(config.mutation_rate)).shape[0]
    is_batched = num_rates > 1

    test_data_shape = (
      max(config.memory_config.auto_tuning_config.probe_batch_sizes),
      len(config.seed_sequence),
    )
    if is_batched:
      test_data_shape = (num_rates, *test_data_shape)

    test_data = (
      tune_key,
      jnp.zeros(test_data_shape, dtype=jnp.int8),
    )
    batch_size = auto_tune_batch_size(
      func=fitness_fn,
      test_data=test_data,
      config=config.memory_config.auto_tuning_config,
    )
    logger.info("Auto-tuning selected chunk size: %d", batch_size)

    # Re-create the fitness function with the optimal chunk size
    fitness_fn = get_fitness_function(
      evaluator_config=config.fitness_evaluator,
      n_states=config.n_states,
      translate_func=translate_func,
      batch_size=batch_size,
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

  if isinstance(config.seed_sequence, jax.Array):
    seed_sequences = config.seed_sequence
  else:
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
            strict=False,
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
  sequence_type_inputs = _convert_to_list(config.sequence_type)
  return jax_inputs, sequence_type_inputs


def run_experiment(config: BaseSamplerConfig, output_dir: str | Path, seed: int = 0) -> None:
  """Run a sampling experiment based on the provided configuration."""
  key = jax.random.PRNGKey(seed)
  output_path = Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)
  run_uuid = uuid.uuid4()
  logger.info("Run UUID: %s", run_uuid)
  # Create metadata file with run configuration and git commit hash
  create_metadata_file(config, output_path)

  sampler_def = _validate_config(config)
  run_fn: Callable[..., tuple[Any, dict[str, Any]]] = sampler_def["run_fn"]
  key, fitness_fn = _setup_fitness_function(key, config)
  mutation_fn = _setup_mutation_function(config)
  annealing_fn = (
    get_annealing_function(config.annealing_config)
    if hasattr(config, "annealing_config") and config.annealing_config is not None
    else None
  )
  writer, io_callback = _setup_writer_callback(output_path / f"data_{run_uuid}.arrayrecord")
  try:
    logger.info(
      "Starting run of type '%s'...",
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
        mutation_rate=jax_inputs.get("mutation_rate", None),
      )
      for stype in sequence_type_inputs
    )

    logger.info("Starting sampler loop...")
    # Materialize initial states (one per sequence_type). For now we run the
    # sampler for the first sequence type only (most experiments use a single
    # sequence type). The per-sampler run functions expect explicit args
    # rather than a single config object.
    initial_states_list = list(initial_states)

    if config.sampler_type == "smc":
      # run_smc_loop expects: num_samples, algorithm, resampling_approach,
      # initial_state, fitness_fn, mutation_fn, annealing_fn, writer_callback
      num_samples = (
        config.num_samples if isinstance(config.num_samples, int) else config.num_samples[0]
      )
      # Type narrowing: at this point we know config is SMCConfig
      algorithm = config.algorithm  # type: ignore[attr-defined]
      resampling_approach = config.resampling_approach  # type: ignore[attr-defined]
      initial_state = initial_states_list[0]

      final_state, _ = run_fn(
        num_samples,
        algorithm,
        resampling_approach,
        initial_state,
        fitness_fn,
        mutation_fn,
        annealing_fn,
        io_callback,
      )
    else:
      # Fallback: call the run function with commonly used kwargs for other
      # sampler implementations. This preserves backward compatibility with
      # the previous invocation style used in tests/mocks.
      final_state, _ = run_fn(
        config=config,
        initial_state=initial_states_list,
        fitness_fn=fitness_fn,
        mutation_fn=mutation_fn,
        annealing_fn=annealing_fn,
        io_callback=io_callback,
      )
    jax.block_until_ready(final_state)
    logger.info("Sampler loop finished.")

    # 6. Write results to disk

    logger.info("Run completed successfully.")
  finally:
    writer.close()
    logger.info("Output writer closed.")
