"""Tests for NUTS sampling algorithm."""

from __future__ import annotations

import pytest

from proteinsmc.models.nuts import NUTSConfig


class TestNUTSConfig:
  """Test the NUTSConfig configuration class."""

  def test_config_creation(self, fitness_evaluator_mock) -> None:
    """Test NUTS config creation.

    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.

    Returns:
        None

    Raises:
        AssertionError: If config initialization fails.

    Example:
        >>> test_config_creation(fitness_evaluator_mock)

    """
    config = NUTSConfig(
      num_samples=10,
      mutation_rate=0.1,
      fitness_evaluator=fitness_evaluator_mock,
      seed_sequence="ACDEF",
      prng_seed=42,
    )

    assert config.num_samples == 10
    assert config.mutation_rate == 0.1

  def test_different_num_samples(self, fitness_evaluator_mock) -> None:
    """Test NUTS config with different numbers of samples.

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
      config = NUTSConfig(
        num_samples=num_samples,
        mutation_rate=0.1,
        fitness_evaluator=fitness_evaluator_mock,
        seed_sequence="ACDEF",
        prng_seed=42,
      )

      assert config.num_samples == num_samples

  def test_frozen_config(self, fitness_evaluator_mock) -> None:
    """Test that NUTSConfig is frozen and immutable.

    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.

    Returns:
        None

    Raises:
        AttributeError: If config is mutable.

    Example:
        >>> test_frozen_config(fitness_evaluator_mock)

    """
    config = NUTSConfig(
      num_samples=10,
      mutation_rate=0.1,
      fitness_evaluator=fitness_evaluator_mock,
      seed_sequence="ACDEF",
      prng_seed=42,
    )

    with pytest.raises((AttributeError, Exception)):  # noqa: B017,PT011
      config.num_samples = 20  # type: ignore[misc]
