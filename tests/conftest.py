import pytest
import jax
from typing import Tuple
from proteinsmc.models import SMCConfig, FitnessEvaluator, FitnessFunction

import jax.numpy as jnp


@pytest.fixture(scope="session")
def prng_key() -> jax.Array:
  """Provides a seeded JAX PRNG key for reproducible tests.
  Args:
    None
  Returns:
    jax.Array: Seeded PRNG key.
  Example:
    >>> key = prng_key()
    >>> jax.random.uniform(key)
  """
  return jax.random.PRNGKey(42)

@pytest.fixture(scope="session")
def sample_sequences() -> Tuple[str, str]:
  """Provides sample protein and nucleotide sequences for testing.
  Args:
    None
  Returns:
    Tuple[str, str]: (protein_sequence, nucleotide_sequence)
  Example:
    >>> protein, nucleotide = sample_sequences()
    >>> print(protein, nucleotide)
  """
  protein_sequence = "MKTFFVAGVIL"
  nucleotide_sequence = "ATGAAGACCTTTTTGTTGCTGGAGTTATTCTT"
  return protein_sequence, nucleotide_sequence

@pytest.fixture(scope="session")
def fitness_evaluator_mock() -> FitnessEvaluator:
  """Provides a mock FitnessEvaluator instance for testing purposes."""
  mock_fitness_function = FitnessFunction(name="mock_fitness", n_states=20)
  return FitnessEvaluator(fitness_functions=(mock_fitness_function,))

@pytest.fixture(scope="session")
def default_smc_config(fitness_evaluator_mock: FitnessEvaluator) -> SMCConfig:
  """Provides a basic default SMCConfig object for SMC tests.
  Args:
    None
  Returns:
    SMCConfig: Default configuration object.
  Example:
    >>> config = default_smc_config()
    >>> print(config.num_particles)
  """
  return SMCConfig(
    prng_seed=42,
    population_size=32,
    num_samples=10,
    mutation_rate=0.01,
    n_states=20,
    sequence_type="protein",
    seed_sequence="MKTFFVAGVIL",
    diversification_ratio=0.1,
    sampler_type="smc",
    algorithm="AdaptiveTemperedSMC",
    fitness_evaluator=fitness_evaluator_mock,
  )