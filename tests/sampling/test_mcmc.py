"""Tests for MCMC sampling algorithm."""

from __future__ import annotations

import pytest

from proteinsmc.models.mcmc import DEFAULT_STEP_SIZE, MCMCConfig


class TestMCMCConfig:
  """Test the MCMCConfig configuration class."""

  def test_default_config(self, fitness_evaluator_mock) -> None:
    """Test MCMC config with default parameters.

    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.

    Returns:
        None

    Raises:
        AssertionError: If config initialization fails.

    Example:
        >>> test_default_config(fitness_evaluator_mock)

    """
    config = MCMCConfig(
      num_samples=10,
      mutation_rate=0.1,
      fitness_evaluator=fitness_evaluator_mock,
      seed_sequence="ACDEF",
      prng_seed=42,
    )

    assert config.num_samples == 10
    assert config.mutation_rate == 0.1
    assert config.step_size == DEFAULT_STEP_SIZE

  def test_custom_step_size(self, fitness_evaluator_mock) -> None:
    """Test MCMC config with custom step size.

    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.

    Returns:
        None

    Raises:
        AssertionError: If config initialization fails.

    Example:
        >>> test_custom_step_size(fitness_evaluator_mock)

    """
    config = MCMCConfig(
      num_samples=10,
      mutation_rate=0.1,
      fitness_evaluator=fitness_evaluator_mock,
      seed_sequence="ACDEF",
      prng_seed=42,
      step_size=0.05,
    )

    assert config.step_size == 0.05

  def test_different_num_samples(self, fitness_evaluator_mock) -> None:
    """Test MCMC config with different numbers of samples.

    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.

    Returns:
        None

    Raises:
        AssertionError: If config initialization fails.

    Example:
        >>> test_different_num_samples(fitness_evaluator_mock)

    """
    for num_samples in [1, 10, 100]:
      config = MCMCConfig(
        num_samples=num_samples,
        mutation_rate=0.1,
        fitness_evaluator=fitness_evaluator_mock,
        seed_sequence="ACDEF",
        prng_seed=42,
      )

      assert config.num_samples == num_samples

  def test_different_step_sizes(self, fitness_evaluator_mock) -> None:
    """Test MCMC config with different step sizes.

    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.

    Returns:
        None

    Raises:
        AssertionError: If config initialization fails.

    Example:
        >>> test_different_step_sizes(fitness_evaluator_mock)

    """
    for step_size in [0.01, 0.1, 1.0]:
      config = MCMCConfig(
        num_samples=10,
        mutation_rate=0.1,
        fitness_evaluator=fitness_evaluator_mock,
        seed_sequence="ACDEF",
        prng_seed=42,
        step_size=step_size,
      )

      assert config.step_size == step_size

  def test_frozen_config(self, fitness_evaluator_mock) -> None:
    """Test that MCMCConfig is frozen and immutable.

    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.

    Returns:
        None

    Raises:
        AttributeError: If config is mutable.

    Example:
        >>> test_frozen_config(fitness_evaluator_mock)

    """
    config = MCMCConfig(
      num_samples=10,
      mutation_rate=0.1,
      fitness_evaluator=fitness_evaluator_mock,
      seed_sequence="ACDEF",
      prng_seed=42,
    )

    with pytest.raises((AttributeError, Exception)):  # noqa: B017,PT011
      config.step_size = 0.2  # type: ignore[misc]

  def test_default_step_size_value(self) -> None:
    """Test that DEFAULT_STEP_SIZE is set correctly.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If default value is incorrect.

    Example:
        >>> test_default_step_size_value()

    """
    assert DEFAULT_STEP_SIZE == 1e-1
    assert isinstance(DEFAULT_STEP_SIZE, float)
