"""Main entry point for running experiments."""

import logging

import jax

from proteinsmc.io import RunManager
from proteinsmc.models.parallel_replica import ParallelReplicaConfig
from proteinsmc.models.sampler_base import BaseSamplerConfig
from proteinsmc.models.smc import SMCConfig
from proteinsmc.sampling.smc.parallel_replica import initialize_prsmc_state, run_prsmc_loop
from proteinsmc.sampling.smc.smc import initialize_smc_state, run_smc_loop
from proteinsmc.utils.annealing import get_annealing_function
from proteinsmc.utils.fitness import get_fitness_function

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
}


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_experiment(config: BaseSamplerConfig, output_dir: str, seed: int = 0) -> None:
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

  # 2. Get the appropriate JIT-compatible functions from our factories
  fitness_fn = get_fitness_function(config.fitness_evaluator)
  annealing_fn = get_annealing_function(config.annealing_config)
  initialize_fn = sampler_def["initialize_fn"]
  run_fn = sampler_def["run_fn"]

  with RunManager(output_dir, config) as writer:
    logger.info(
      "Starting run %s of type '%s'...",
      writer.run_id,
      config.sampler_type,
    )

    # 3. Initialize sampler state
    key, init_key = jax.random.split(key)
    initial_state = initialize_fn(config, init_key)

    # 4. Run the core sampler loop
    final_state, all_outputs = run_fn(
      config=config,
      initial_state=initial_state,
      fitness_fn=fitness_fn,
      annealing_fn=annealing_fn,
    )

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
