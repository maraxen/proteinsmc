"""Main entry point for running experiments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

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
from proteinsmc.sampling.gibbs import initialize_gibbs_state
from proteinsmc.sampling.hmc import initialize_hmc_state
from proteinsmc.sampling.mcmc import initialize_mcmc_state
from proteinsmc.sampling.nuts import initialize_nuts_state
from proteinsmc.sampling.particle_systems.parallel_replica import (
  initialize_prsmc_state,
  run_prsmc_loop,
)
from proteinsmc.sampling.particle_systems.smc import initialize_smc_state, run_smc_loop
from proteinsmc.utils.annealing import get_annealing_function
from proteinsmc.utils.constants import NUCLEOTIDES_NUM_STATES
from proteinsmc.utils.fitness import get_fitness_function
from proteinsmc.utils.memory import auto_tune_chunk_size
from proteinsmc.utils.mutation import make_mutation_fn
from proteinsmc.utils.translation import aa_to_nucleotide, nucleotide_to_aa

if TYPE_CHECKING:
  from proteinsmc.models.sampler_base import BaseSamplerConfig

SAMPLER_REGISTRY = {
  "smc": {
    "config_cls": SMCConfig,
    "initialize_fn": initialize_smc_state,
    "run_fn": run_smc_loop,
  },
  "parallel_replica": {
    "config_cls": ParallelReplicaConfig,
    "initialize_fn": initialize_prsmc_state,
    "run_fn": run_prsmc_loop,
  },
  "gibbs": {
    "config_cls": GibbsConfig,
    "initialize_fn": initialize_gibbs_state,
    "run_fn": run_gibbs_loop,
  },
  "mcmc": {
    "config_cls": MCMCConfig,
    "initialize_fn": initialize_mcmc_state,
    "run_fn": run_mcmc_loop,
  },
  "hmc": {
    "config_cls": HMCConfig,
    "initialize_fn": initialize_hmc_state,
    "run_fn": run_hmc_loop,
  },
  "nuts": {
    "config_cls": NUTSConfig,
    "initialize_fn": initialize_nuts_state,
    "run_fn": run_nuts_loop,
  },
}


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_experiment(config: BaseSamplerConfig, output_dir: str | Path, seed: int = 0) -> None:
  """Run a sampling experiment based on the provided configuration.

  Args:
      config: A configuration object inheriting from BaseSamplerConfig.
      output_dir: The directory to save the results.
      seed: An integer seed for the random number generator.

  """
  key = jax.random.PRNGKey(seed)

  if config.sampler_type not in SAMPLER_REGISTRY:
    msg = (
      f"Unknown sampler_type: '{config.sampler_type}'. "
      f"Available types: {list(SAMPLER_REGISTRY.keys())}."
    )
    raise ValueError(msg)

  sampler_def = SAMPLER_REGISTRY[config.sampler_type]

  # 1. Validate config type
  if not isinstance(config, sampler_def["config_cls"]):
    msg = (
      f"Configuration object of type {type(config)} does not match "
      f"sampler type '{config.sampler_type}' which requires {sampler_def['config_cls']}."
    )
    raise TypeError(msg)

  chunk_size = None
  fitness_fn = get_fitness_function(
    evaluator_config=config.fitness_evaluator,
    n_states=config.n_states,
    translate_func=nucleotide_to_aa
    if config.n_states == NUCLEOTIDES_NUM_STATES
    else aa_to_nucleotide,
  )
  if config.memory_config.auto_tuning_config.enable_auto_tuning:
    logger.info(
      "Auto-tuning is enabled. This may affect performance and memory usage. "
      "Ensure that the auto-tuning configuration is set correctly.",
    )
    chunk_size = auto_tune_chunk_size(
      func=fitness_fn,
      test_data=(
        jax.random.split(key)[0],
        jnp.zeros(
          (
            max(config.memory_config.auto_tuning_config.probe_chunk_sizes),
            len(config.seed_sequence),
          ),
        ),
        None,
      ),
      config=config.memory_config.auto_tuning_config,
      static_args={"sequence_type": config.sequence_type},
    )
    fitness_fn = get_fitness_function(
      evaluator_config=config.fitness_evaluator,
      n_states=config.n_states,
      translate_func=nucleotide_to_aa
      if config.n_states == NUCLEOTIDES_NUM_STATES
      else aa_to_nucleotide,
      chunk_size=chunk_size,
    )

  initialize_fn = sampler_def["initialize_fn"]
  run_fn = sampler_def["run_fn"]
  mutation_fn = make_mutation_fn(
    config=config,
  )

  with RunManager(Path(output_dir), config) as writer:
    logger.info(
      "Starting run %s of type '%s'...",
      writer.run_id,
      config.sampler_type,
    )

    # 3. Initialize sampler state
    key, init_key = jax.random.split(key)
    initial_state = initialize_fn(config, fitness_fn, init_key)

    if hasattr(config, "annealing_config"):
      annealing_fn = get_annealing_function(config.annealing_config)  # type: ignore[assignment]
      final_state, all_outputs = run_fn(
        config=config,
        initial_state=initial_state,
        log_prob_fn=fitness_fn,
        mutation_fn=mutation_fn,
        annealing_fn=annealing_fn,
      )
    else:
      final_state, all_outputs = run_fn(
        config=config,
        initial_state=initial_state,
        log_prob_fn=fitness_fn,
        mutation_fn=mutation_fn,
      )

    # 4. Run the core sampler loop

    # 5. Write results to disk
    logger.info("Writing results to disk...")
    num_steps = all_outputs[next(iter(all_outputs))].shape[0]
    for i in range(num_steps):
      step_output_tree = jax.tree_util.tree_map(lambda x: x[i], all_outputs)

      scalar_metrics: dict[str, int | float] = {"step": i}
      for metric_name, metric_value in step_output_tree.items():
        if metric_value.ndim == 0:
          scalar_metrics[metric_name] = float(metric_value)
      writer.log_scalars(scalar_metrics)

      writer.step(step_output_tree)

  logger.info("Run %s completed successfully.", writer.run_id)
