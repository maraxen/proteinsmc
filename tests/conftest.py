"""Common fixtures and utilities for testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import chex
import jax
import jax.numpy as jnp
import pytest

if TYPE_CHECKING:
  from jaxtyping import PRNGKeyArray

from proteinsmc.models import (
  AnnealingConfig,
  AutoTuningConfig,
  FitnessEvaluator,
  FitnessFunction,
  MemoryConfig,
  SMCConfig,
)


@pytest.fixture
def rng_key() -> PRNGKeyArray:
  """Provide a consistent PRNG key for testing."""
  return jax.random.PRNGKey(42)


@pytest.fixture(scope="session")
def fitness_evaluator_mock() -> FitnessEvaluator:
  """Provides a mock FitnessEvaluator instance for testing purposes."""
  mock_fitness_function = FitnessFunction(name="mock_fitness", n_states=20)
  return FitnessEvaluator(fitness_functions=(mock_fitness_function,))

@pytest.fixture
def sample_nucleotide_sequence() -> jnp.ndarray:
  """Provide a sample nucleotide sequence for testing."""
  # Simple 12-nucleotide sequence (4 codons)
  return jnp.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=jnp.int8)


@pytest.fixture
def sample_protein_sequence() -> jnp.ndarray:
  """Provide a sample protein sequence for testing."""
  # Simple 4-amino acid sequence
  return jnp.array([0, 1, 2, 3], dtype=jnp.int8)


@pytest.fixture
def sample_population_nucleotides() -> jnp.ndarray:
  """Provide a sample population of nucleotide sequences."""
  # Population of 8 sequences, each 12 nucleotides long
  return jnp.array([
    [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
    [1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0],
    [2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
    [3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2],
    [0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1],
    [1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2],
    [2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 3, 3],
    [3, 3, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0],
  ], dtype=jnp.int8)


@pytest.fixture
def sample_population_proteins() -> jnp.ndarray:
  """Provide a sample population of protein sequences."""
  # Population of 8 sequences, each 4 amino acids long
  return jnp.array([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15],
    [16, 17, 18, 19],
    [0, 5, 10, 15],
    [1, 6, 11, 16],
    [2, 7, 12, 17],
  ], dtype=jnp.int8)


@pytest.fixture
def basic_memory_config() -> MemoryConfig:
  """Provide a basic memory configuration for testing."""
  return MemoryConfig(
    batch_size=16,
    enable_batched_computation=True,
    device_memory_fraction=0.8,
    auto_tuning_config=AutoTuningConfig(
      enable_auto_tuning=False,
      probe_chunk_sizes=(8, 16, 32),
      max_probe_iterations=2,
      memory_safety_factor=0.8,
      performance_tolerance=0.1,
    ),
  )


@pytest.fixture
def basic_annealing_config() -> AnnealingConfig:
  """Provide a basic annealing configuration for testing."""
  return AnnealingConfig(
    annealing_fn="linear",
    beta_max=1.0,
    n_steps=10,
    kwargs={},
  )


@pytest.fixture
def basic_fitness_evaluator() -> FitnessEvaluator:
  """Provide a basic fitness evaluator for testing."""
  return FitnessEvaluator(
    fitness_functions=(
      FitnessFunction(name="test_function", n_states=20),
    ),
  )


@pytest.fixture
def basic_smc_config(
  basic_memory_config: MemoryConfig,
  basic_annealing_config: AnnealingConfig,
  basic_fitness_evaluator: FitnessEvaluator,
) -> SMCConfig:
  """Provide a basic SMC configuration for testing."""
  return SMCConfig(
    prng_seed=42,
    seed_sequence="MKAF",
    num_samples=10,
    n_states=20,
    mutation_rate=0.1,
    diversification_ratio=0.5,
    sequence_type="protein",
    fitness_evaluator=basic_fitness_evaluator,
    memory_config=basic_memory_config,
    annealing_config=basic_annealing_config,
    population_size=16,
  )


@pytest.fixture
def sample_log_weights() -> jnp.ndarray:
  """Provide sample log weights for testing resampling."""
  weights = jnp.array([0.1, 0.2, 0.3, 0.15, 0.05, 0.1, 0.05, 0.05])
  return jnp.log(weights)


@pytest.fixture
def sample_fitness_scores() -> jnp.ndarray:
  """Provide sample fitness scores for testing."""
  return jnp.array([1.5, 2.0, 1.8, 2.2, 1.3, 1.9, 1.7, 2.1])


class DummyFitnessEvaluator(FitnessEvaluator):
  pass

class DummyMemoryConfig(MemoryConfig):
  pass

class DummyAnnealingConfig(AnnealingConfig):
  pass

@pytest.fixture
def valid_config_kwargs(monkeypatch):
  """Fixture providing valid arguments for BaseSamplerConfig.

  Args:
    monkeypatch: pytest monkeypatch fixture for patching methods.

  Returns:
    dict: Valid arguments for BaseSamplerConfig instantiation.

  Example:
    >>> config = BaseSamplerConfig(**valid_config_kwargs)
  """
  # Monkeypatch __post_init__ methods to avoid validation/initialization
  monkeypatch.setattr(FitnessEvaluator, "__post_init__", lambda self: None)

  return dict(
    prng_seed=123,
    sampler_type="smc",
    seed_sequence="ACDEFGHIKLMNPQRSTVWY",
    num_samples=10,
    n_states=20,
    mutation_rate=0.05,
    diversification_ratio=0.2,
    sequence_type="protein",
    algorithm="AdaptiveTemperedSMC",
    fitness_evaluator=DummyFitnessEvaluator(
      fitness_functions=()),
    memory_config=DummyMemoryConfig(),
    annealing_config=DummyAnnealingConfig(
      annealing_fn="dummy_fn",
      beta_max=1.0,
      n_steps=100,
    ),
  )