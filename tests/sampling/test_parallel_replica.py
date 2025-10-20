"""Tests for Parallel Replica SMC configuration and structures."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from proteinsmc.models.parallel_replica import (
  ExchangeConfig,
  ParallelReplicaConfig,
  PRSMCOutput,
)


class TestExchangeConfig:
  """Test the ExchangeConfig configuration class."""

  def test_default_config(self, fitness_evaluator_mock, annealing_config_mock) -> None:
    """Test ExchangeConfig with default parameters.

    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
        annealing_config_mock: Fixture providing mock annealing config.

    Returns:
        None

    Raises:
        AssertionError: If config initialization fails.

    Example:
        >>> test_default_config(fitness_evaluator_mock, annealing_config_mock)

    """
    config = ExchangeConfig(
      fitness_evaluator=fitness_evaluator_mock,
      meta_annealing_schedule=annealing_config_mock,
    )

    assert config.n_islands == 1
    assert config.population_size_per_island == 64
    assert config.n_exchange_attempts == 10
    assert config.exchange_frequency == 5
    assert config.sequence_type == "protein"
    assert not config.track_lineage

  def test_custom_n_islands(self, fitness_evaluator_mock, annealing_config_mock) -> None:
    """Test ExchangeConfig with custom n_islands.

    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
        annealing_config_mock: Fixture providing mock annealing config.

    Returns:
        None

    Raises:
        AssertionError: If config initialization fails.

    Example:
        >>> test_custom_n_islands(fitness_evaluator_mock, annealing_config_mock)

    """
    config = ExchangeConfig(
      n_islands=4,
      fitness_evaluator=fitness_evaluator_mock,
      meta_annealing_schedule=annealing_config_mock,
    )

    assert config.n_islands == 4

  def test_custom_population_size(
    self,
    fitness_evaluator_mock,
    annealing_config_mock,
  ) -> None:
    """Test ExchangeConfig with custom population size.

    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
        annealing_config_mock: Fixture providing mock annealing config.

    Returns:
        None

    Raises:
        AssertionError: If config initialization fails.

    Example:
        >>> test_custom_population_size(fitness_evaluator_mock, annealing_config_mock)

    """
    config = ExchangeConfig(
      population_size_per_island=128,
      fitness_evaluator=fitness_evaluator_mock,
      meta_annealing_schedule=annealing_config_mock,
    )

    assert config.population_size_per_island == 128

  def test_custom_exchange_params(
    self,
    fitness_evaluator_mock,
    annealing_config_mock,
  ) -> None:
    """Test ExchangeConfig with custom exchange parameters.

    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
        annealing_config_mock: Fixture providing mock annealing config.

    Returns:
        None

    Raises:
        AssertionError: If config initialization fails.

    Example:
        >>> test_custom_exchange_params(fitness_evaluator_mock, annealing_config_mock)

    """
    config = ExchangeConfig(
      n_exchange_attempts=20,
      exchange_frequency=10,
      fitness_evaluator=fitness_evaluator_mock,
      meta_annealing_schedule=annealing_config_mock,
    )

    assert config.n_exchange_attempts == 20
    assert config.exchange_frequency == 10

  def test_sequence_type_protein(
    self,
    fitness_evaluator_mock,
    annealing_config_mock,
  ) -> None:
    """Test ExchangeConfig with protein sequence type.

    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
        annealing_config_mock: Fixture providing mock annealing config.

    Returns:
        None

    Raises:
        AssertionError: If config initialization fails.

    Example:
        >>> test_sequence_type_protein(fitness_evaluator_mock, annealing_config_mock)

    """
    config = ExchangeConfig(
      sequence_type="protein",
      fitness_evaluator=fitness_evaluator_mock,
      meta_annealing_schedule=annealing_config_mock,
    )

    assert config.sequence_type == "protein"

  def test_sequence_type_nucleotide(
    self,
    fitness_evaluator_mock,
    annealing_config_mock,
  ) -> None:
    """Test ExchangeConfig with nucleotide sequence type.

    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
        annealing_config_mock: Fixture providing mock annealing config.

    Returns:
        None

    Raises:
        AssertionError: If config initialization fails.

    Example:
        >>> test_sequence_type_nucleotide(fitness_evaluator_mock, annealing_config_mock)

    """
    config = ExchangeConfig(
      sequence_type="nucleotide",
      fitness_evaluator=fitness_evaluator_mock,
      meta_annealing_schedule=annealing_config_mock,
    )

    assert config.sequence_type == "nucleotide"

  def test_track_lineage_enabled(
    self,
    fitness_evaluator_mock,
    annealing_config_mock,
  ) -> None:
    """Test ExchangeConfig with lineage tracking enabled.

    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
        annealing_config_mock: Fixture providing mock annealing config.

    Returns:
        None

    Raises:
        AssertionError: If config initialization fails.

    Example:
        >>> test_track_lineage_enabled(fitness_evaluator_mock, annealing_config_mock)

    """
    config = ExchangeConfig(
      track_lineage=True,
      fitness_evaluator=fitness_evaluator_mock,
      meta_annealing_schedule=annealing_config_mock,
    )

    assert config.track_lineage is True


class TestParallelReplicaConfig:
  """Test the ParallelReplicaConfig configuration class."""

  def test_config_creation(
    self,
    smc_config_mock,
    fitness_evaluator_mock,
    annealing_config_mock,
  ) -> None:
    """Test ParallelReplicaConfig creation.

    Args:
        smc_config_mock: Fixture providing mock SMC config.
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
        annealing_config_mock: Fixture providing mock annealing config.

    Returns:
        None

    Raises:
        AssertionError: If config initialization fails.

    Example:
        >>> test_config_creation(smc_config_mock, fitness_evaluator_mock, annealing_config_mock)

    """
    config = ParallelReplicaConfig(
      num_samples=10,
      mutation_rate=0.1,
      fitness_evaluator=fitness_evaluator_mock,
      seed_sequence="ACDEF",
      prng_seed=42,
      smc_config=smc_config_mock,
      n_islands=4,
      island_betas=[0.25, 0.5, 0.75, 1.0],
      meta_annealing_schedule=annealing_config_mock,
    )

    assert config.num_samples == 10
    assert config.n_islands == 4
    assert len(config.island_betas) == 4
    assert config.sampler_type == "parallel_replica"

  def test_config_validation_n_islands(
    self,
    smc_config_mock,
    fitness_evaluator_mock,
    annealing_config_mock,
  ) -> None:
    """Test ParallelReplicaConfig validation for n_islands.

    Args:
        smc_config_mock: Fixture providing mock SMC config.
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
        annealing_config_mock: Fixture providing mock annealing config.

    Returns:
        None

    Raises:
        ValueError: If n_islands is invalid.

    Example:
        >>> test_config_validation_n_islands(smc_config_mock, fitness_evaluator_mock, annealing_config_mock)

    """
    with pytest.raises(ValueError, match="n_islands must be positive"):
      ParallelReplicaConfig(
        num_samples=10,
        mutation_rate=0.1,
        fitness_evaluator=fitness_evaluator_mock,
        seed_sequence="ACDEF",
        prng_seed=42,
        smc_config=smc_config_mock,
        n_islands=0,
        island_betas=[],
        meta_annealing_schedule=annealing_config_mock,
      )

  def test_config_validation_exchange_frequency(
    self,
    smc_config_mock,
    fitness_evaluator_mock,
    annealing_config_mock,
  ) -> None:
    """Test ParallelReplicaConfig validation for exchange frequency.

    Args:
        smc_config_mock: Fixture providing mock SMC config.
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
        annealing_config_mock: Fixture providing mock annealing config.

    Returns:
        None

    Raises:
        ValueError: If exchange frequency is invalid.

    Example:
        >>> test_config_validation_exchange_frequency(smc_config_mock, fitness_evaluator_mock, annealing_config_mock)

    """
    with pytest.raises(ValueError, match="exchange_frequency must be positive"):
      ParallelReplicaConfig(
        num_samples=10,
        mutation_rate=0.1,
        fitness_evaluator=fitness_evaluator_mock,
        seed_sequence="ACDEF",
        prng_seed=42,
        smc_config=smc_config_mock,
        n_islands=2,
        island_betas=[0.5, 1.0],
        exchange_frequency=0,  # Invalid exchange frequency
        meta_annealing_schedule=annealing_config_mock,
      )

  def test_config_validation_island_betas_mismatch(
    self,
    smc_config_mock,
    fitness_evaluator_mock,
    annealing_config_mock,
  ) -> None:
    """Test ParallelReplicaConfig validation for island_betas mismatch.

    Args:
        smc_config_mock: Fixture providing mock SMC config.
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
        annealing_config_mock: Fixture providing mock annealing config.

    Returns:
        None

    Raises:
        ValueError: If island_betas count doesn't match n_islands.

    Example:
        >>> test_config_validation_island_betas_mismatch(smc_config_mock, fitness_evaluator_mock, annealing_config_mock)

    """
    with pytest.raises(ValueError, match="island_betas must match n_islands"):
      ParallelReplicaConfig(
        num_samples=10,
        mutation_rate=0.1,
        fitness_evaluator=fitness_evaluator_mock,
        seed_sequence="ACDEF",
        prng_seed=42,
        smc_config=smc_config_mock,
        n_islands=4,
        island_betas=[0.5, 1.0],  # Only 2 betas for 4 islands
        meta_annealing_schedule=annealing_config_mock,
      )


class TestPRSMCOutput:
  """Test the PRSMCOutput data structure."""

  def test_output_creation(self) -> None:
    """Test PRSMCOutput creation.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If output creation fails.

    Example:
        >>> test_output_creation()

    """
    output = PRSMCOutput(
      ess_per_island=jnp.array([10.0, 20.0, 30.0]),
      mean_fitness_per_island=jnp.array([0.5, 0.7, 0.9]),
      max_fitness_per_island=jnp.array([0.8, 0.9, 1.0]),
      logZ_increment_per_island=jnp.array([0.1, 0.2, 0.3]),
      lineage_per_island=None,
      meta_beta=jnp.array([0.25, 0.5, 0.75]),
      num_accepted_swaps=jnp.array([5]),
      num_attempted_swaps=jnp.array([10]),
    )

    assert len(output.ess_per_island) == 3
    assert len(output.mean_fitness_per_island) == 3
    assert len(output.max_fitness_per_island) == 3

  def test_output_field_values(self) -> None:
    """Test PRSMCOutput field values.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If output values are incorrect.

    Example:
        >>> test_output_field_values()

    """
    ess = jnp.array([15.0, 25.0])
    mean_fitness = jnp.array([0.6, 0.8])
    max_fitness = jnp.array([0.85, 0.95])
    logZ_increment = jnp.array([0.15, 0.25])
    meta_beta = jnp.array([0.5, 1.0])
    num_accepted = jnp.array([3])
    num_attempted = jnp.array([8])

    output = PRSMCOutput(
      ess_per_island=ess,
      mean_fitness_per_island=mean_fitness,
      max_fitness_per_island=max_fitness,
      logZ_increment_per_island=logZ_increment,
      lineage_per_island=None,
      meta_beta=meta_beta,
      num_accepted_swaps=num_accepted,
      num_attempted_swaps=num_attempted,
    )

    assert jnp.array_equal(output.ess_per_island, ess)
    assert jnp.array_equal(output.mean_fitness_per_island, mean_fitness)
    assert jnp.array_equal(output.max_fitness_per_island, max_fitness)

  def test_output_type_name(self) -> None:
    """Test PRSMCOutput type name property.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If type name is incorrect.

    Example:
        >>> test_output_type_name()

    """
    output = PRSMCOutput(
      ess_per_island=jnp.array([10.0]),
      mean_fitness_per_island=jnp.array([0.5]),
      max_fitness_per_island=jnp.array([0.8]),
      logZ_increment_per_island=jnp.array([0.1]),
      lineage_per_island=None,
      meta_beta=jnp.array([1.0]),
      num_accepted_swaps=jnp.array([5]),
      num_attempted_swaps=jnp.array([10]),
    )

    assert output.output_type_name == "ParallelReplicaSMC"
