"""Advanced tests for ParallelReplicaConfig validation."""

from __future__ import annotations

import pytest

from proteinsmc.models.annealing import AnnealingConfig
from proteinsmc.models.parallel_replica import PRSMCOutput, ParallelReplicaConfig
from proteinsmc.models.smc import SMCConfig


class TestParallelReplicaConfigValidation:
  """Test validation logic in ParallelReplicaConfig."""

  def test_negative_n_islands(self, fitness_evaluator_mock, basic_smc_config) -> None:
    """Test error when n_islands is negative.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
        basic_smc_config: Fixture providing basic SMC config.
    
    Returns:
        None
    
    Raises:
        ValueError: If n_islands is negative.
    
    Example:
        >>> test_negative_n_islands(fitness_evaluator_mock, basic_smc_config)
    
    """
    annealing_config = AnnealingConfig(annealing_fn="linear", n_steps=10)

    with pytest.raises(ValueError, match="n_islands must be positive"):
      ParallelReplicaConfig(
        n_islands=-1,
        exchange_frequency=5,
        fitness_evaluator=fitness_evaluator_mock,
        smc_config=basic_smc_config,
        meta_annealing_schedule=annealing_config,
        island_betas=[-0.5],
      )

  def test_zero_n_islands(self, fitness_evaluator_mock, basic_smc_config) -> None:
    """Test error when n_islands is zero.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
        basic_smc_config: Fixture providing basic SMC config.
    
    Returns:
        None
    
    Raises:
        ValueError: If n_islands is zero.
    
    Example:
        >>> test_zero_n_islands(fitness_evaluator_mock, basic_smc_config)
    
    """
    annealing_config = AnnealingConfig(annealing_fn="linear", n_steps=10)

    with pytest.raises(ValueError, match="n_islands must be positive"):
      ParallelReplicaConfig(
        n_islands=0,
        exchange_frequency=5,
        fitness_evaluator=fitness_evaluator_mock,
        smc_config=basic_smc_config,
        meta_annealing_schedule=annealing_config,
        island_betas=[],
      )

  def test_negative_exchange_attempts(self, fitness_evaluator_mock, basic_smc_config) -> None:
    """Test error when n_exchange_attempts is negative.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
        basic_smc_config: Fixture providing basic SMC config.
    
    Returns:
        None
    
    Raises:
        ValueError: If n_exchange_attempts is negative.
    
    Example:
        >>> test_negative_exchange_attempts(fitness_evaluator_mock, basic_smc_config)
    
    """
    annealing_config = AnnealingConfig(annealing_fn="linear", n_steps=10)

    with pytest.raises(ValueError, match="n_exchange_attempts must be non-negative"):
      ParallelReplicaConfig(
        n_islands=2,
        n_exchange_attempts=-1,
        exchange_frequency=5,
        fitness_evaluator=fitness_evaluator_mock,
        smc_config=basic_smc_config,
        meta_annealing_schedule=annealing_config,
        island_betas=[0.5, 1.0],
      )

  def test_zero_exchange_frequency(self, fitness_evaluator_mock, basic_smc_config) -> None:
    """Test error when exchange_frequency is zero.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
        basic_smc_config: Fixture providing basic SMC config.
    
    Returns:
        None
    
    Raises:
        ValueError: If exchange_frequency is zero.
    
    Example:
        >>> test_zero_exchange_frequency(fitness_evaluator_mock, basic_smc_config)
    
    """
    annealing_config = AnnealingConfig(annealing_fn="linear", n_steps=10)

    with pytest.raises(ValueError, match="exchange_frequency must be positive"):
      ParallelReplicaConfig(
        n_islands=2,
        exchange_frequency=0,
        fitness_evaluator=fitness_evaluator_mock,
        smc_config=basic_smc_config,
        meta_annealing_schedule=annealing_config,
        island_betas=[0.5, 1.0],
      )

  def test_mismatched_island_betas(self, fitness_evaluator_mock, basic_smc_config) -> None:
    """Test error when island_betas length doesn't match n_islands.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
        basic_smc_config: Fixture providing basic SMC config.
    
    Returns:
        None
    
    Raises:
        ValueError: If island_betas length doesn't match n_islands.
    
    Example:
        >>> test_mismatched_island_betas(fitness_evaluator_mock, basic_smc_config)
    
    """
    annealing_config = AnnealingConfig(annealing_fn="linear", n_steps=10)

    with pytest.raises(ValueError, match="number of island_betas must match n_islands"):
      ParallelReplicaConfig(
        n_islands=4,
        exchange_frequency=5,
        fitness_evaluator=fitness_evaluator_mock,
        smc_config=basic_smc_config,
        meta_annealing_schedule=annealing_config,
        island_betas=[0.25, 0.5, 0.75],  # Only 3 betas for 4 islands
      )

  def test_valid_config(self, fitness_evaluator_mock, basic_smc_config) -> None:
    """Test valid ParallelReplicaConfig creation.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
        basic_smc_config: Fixture providing basic SMC config.
    
    Returns:
        None
    
    Raises:
        AssertionError: If valid config fails.
    
    Example:
        >>> test_valid_config(fitness_evaluator_mock, basic_smc_config)
    
    """
    annealing_config = AnnealingConfig(annealing_fn="linear", n_steps=10)

    config = ParallelReplicaConfig(
      n_islands=4,
      n_exchange_attempts=10,
      exchange_frequency=5,
      fitness_evaluator=fitness_evaluator_mock,
      smc_config=basic_smc_config,
      meta_annealing_schedule=annealing_config,
      island_betas=[0.25, 0.5, 0.75, 1.0],
    )

    assert config.n_islands == 4
    assert config.n_exchange_attempts == 10
    assert config.exchange_frequency == 5
    assert len(config.island_betas) == 4


class TestPRSMCOutputMetrics:
  """Test PRSMCOutput metric mappings."""

  def test_per_gen_stats_metrics(self) -> None:
    """Test per-generation statistics metric mappings.
    
    Args:
        None
    
    Returns:
        None
    
    Raises:
        AssertionError: If metric mappings are incorrect.
    
    Example:
        >>> test_per_gen_stats_metrics()
    
    """
    import jax.numpy as jnp

    output = PRSMCOutput(
      ess_per_island=jnp.array([1.0]),
      mean_fitness_per_island=jnp.array([1.0]),
      max_fitness_per_island=jnp.array([1.0]),
      logZ_increment_per_island=jnp.array([1.0]),
      meta_beta=jnp.array([1.0]),
      num_accepted_swaps=jnp.array([1]),
      num_attempted_swaps=jnp.array([1]),
    )

    metrics = output.per_gen_stats_metrics
    
    assert "ess_per_island" in metrics
    assert "mean_fitness_per_island" in metrics
    assert "max_fitness_per_island" in metrics
    assert "logZ_increment_per_island" in metrics
    assert "meta_beta" in metrics
    assert "num_accepted_swaps" in metrics
    assert "num_attempted_swaps" in metrics

  def test_output_type_name(self) -> None:
    """Test output type name.
    
    Args:
        None
    
    Returns:
        None
    
    Raises:
        AssertionError: If output type name is incorrect.
    
    Example:
        >>> test_output_type_name()
    
    """
    import jax.numpy as jnp

    output = PRSMCOutput(
      ess_per_island=jnp.array([1.0]),
      mean_fitness_per_island=jnp.array([1.0]),
      max_fitness_per_island=jnp.array([1.0]),
      logZ_increment_per_island=jnp.array([1.0]),
      meta_beta=jnp.array([1.0]),
      num_accepted_swaps=jnp.array([1]),
      num_attempted_swaps=jnp.array([1]),
    )

    assert output.output_type_name == "ParallelReplicaSMC"


class TestExchangeConfig:
  """Test ExchangeConfig dataclass."""

  def test_exchange_config_defaults(self, fitness_evaluator_mock) -> None:
    """Test ExchangeConfig with default values.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
    
    Returns:
        None
    
    Raises:
        AssertionError: If defaults are incorrect.
    
    Example:
        >>> test_exchange_config_defaults(fitness_evaluator_mock)
    
    """
    from proteinsmc.models.parallel_replica import ExchangeConfig

    annealing_config = AnnealingConfig(annealing_fn="linear", n_steps=10)

    config = ExchangeConfig(
      fitness_evaluator=fitness_evaluator_mock,
      meta_annealing_schedule=annealing_config,
    )

    assert config.n_islands == 1
    assert config.population_size_per_island == 64
    assert config.n_exchange_attempts == 10
    assert config.exchange_frequency == 5
    assert config.sequence_type == "protein"

  def test_exchange_config_custom_values(self, fitness_evaluator_mock) -> None:
    """Test ExchangeConfig with custom values.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
    
    Returns:
        None
    
    Raises:
        AssertionError: If custom values not set correctly.
    
    Example:
        >>> test_exchange_config_custom_values(fitness_evaluator_mock)
    
    """
    from proteinsmc.models.parallel_replica import ExchangeConfig

    annealing_config = AnnealingConfig(annealing_fn="linear", n_steps=10)

    config = ExchangeConfig(
      n_islands=8,
      population_size_per_island=128,
      n_exchange_attempts=20,
      exchange_frequency=10,
      fitness_evaluator=fitness_evaluator_mock,
      sequence_type="nucleotide",
      meta_annealing_schedule=annealing_config,
    )

    assert config.n_islands == 8
    assert config.population_size_per_island == 128
    assert config.n_exchange_attempts == 20
    assert config.exchange_frequency == 10
    assert config.sequence_type == "nucleotide"
