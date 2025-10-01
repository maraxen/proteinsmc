"""Unit tests for ParallelReplicaConfig data model.

Tests cover initialization and edge cases for ParallelReplicaConfig.
"""
import pytest
from proteinsmc.models import (
    parallel_replica,
    SMCConfig,
    AnnealingConfig,
)


def test_parallel_replica_config_initialization(
    basic_smc_config: SMCConfig, basic_annealing_config: AnnealingConfig
):
  """Test ParallelReplicaConfig initialization with valid arguments."""
  n_islands = 4
  config = parallel_replica.ParallelReplicaConfig(
      prng_seed=basic_smc_config.prng_seed,
      seed_sequence=basic_smc_config.seed_sequence,
      num_samples=basic_smc_config.num_samples,
      n_states=basic_smc_config.n_states,
      mutation_rate=basic_smc_config.mutation_rate,
      diversification_ratio=basic_smc_config.diversification_ratio,
      sequence_type=basic_smc_config.sequence_type,
      fitness_evaluator=basic_smc_config.fitness_evaluator,
      memory_config=basic_smc_config.memory_config,
      smc_config=basic_smc_config,
      n_islands=n_islands,
      exchange_frequency=5,
      island_betas=[0.25, 0.5, 0.75, 1.0],
      meta_annealing_schedule=basic_annealing_config,
  )
  assert config.n_islands == n_islands
  assert config.exchange_frequency == 5