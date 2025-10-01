"""Unit tests for ParallelReplicaConfig data model.

Tests cover initialization and edge cases for ParallelReplicaConfig.
"""
import pytest
from proteinsmc.models import parallel_replica, FitnessEvaluator, SMCConfig, AnnealingConfig


def test_parallel_replica_config_initialization(
    fitness_evaluator_mock: FitnessEvaluator,
    basic_smc_config: SMCConfig
):
  """Test ParallelReplicaConfig initialization with valid arguments.
  Args:
    fitness_evaluator_mock: A mock fitness evaluator.
    basic_smc_config: A basic SMCConfig.
  Returns:
    None
  Raises:
    AssertionError: If the config fields do not match expected values.
  Example:
    >>> test_parallel_replica_config_initialization(fitness_evaluator_mock, basic_smc_config)
  """
  annealing_config = AnnealingConfig(annealing_fn="linear", n_steps=10)
  n_islands = 4

  config = parallel_replica.ParallelReplicaConfig(
    n_islands=n_islands,
    exchange_frequency=5,
    fitness_evaluator=fitness_evaluator_mock,
    smc_config=basic_smc_config,
    meta_annealing_schedule=annealing_config,
    island_betas=[0.25, 0.5, 0.75, 1.0]
  )
  assert config.n_islands == 4
  assert config.exchange_frequency == 5
  assert config.smc_config == basic_smc_config
  assert config.meta_annealing_schedule == annealing_config