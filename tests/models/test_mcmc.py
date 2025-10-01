"""Unit tests for MCMC sampler data model.

Tests cover initialization and edge cases for MCMC sampler config/model.
"""
import pytest
from proteinsmc.models import mcmc, FitnessEvaluator


def test_mcmc_config_initialization(basic_fitness_evaluator: FitnessEvaluator):
  """Test MCMCConfig initialization with valid arguments."""
  config = mcmc.MCMCConfig(
    num_samples=15,
    step_size=0.05,
    fitness_evaluator=basic_fitness_evaluator,
  )
  assert config.num_samples == 15
  assert config.step_size == 0.05