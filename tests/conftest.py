import pytest
import jax
from typing import Tuple
from src.proteinsmc.sampling.particle_systems.smc import SMCConfig

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
def fitness_evaluator_mock():
  """Provides a mock fitness evaluator for testing purposes.
  Args:
    None
  Returns:
    Callable[[jax.Array], jax.Array]: A mock fitness evaluator function.
  Example:
    >>> evaluator = fitness_evaluator_mock()
    >>> fitness = evaluator(jax.random.uniform(jax.random.PRNGKey(0), (10,)))
  """
  def mock_evaluator(x: jax.Array) -> jax.Array:
    return jnp.sum(x)

  return mock_evaluator

@pytest.fixture(scope="session")
def default_smc_config() -> SMCConfig:
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
    fitness_evaluator=fitness_evaluator_mock(),
  )