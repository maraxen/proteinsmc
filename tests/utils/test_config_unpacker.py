"""Tests for configuration unpacking utilities."""

from __future__ import annotations

import pytest

from proteinsmc.models.fitness import FitnessEvaluator
from proteinsmc.models.sampler_base import BaseSamplerConfig
from proteinsmc.models.smc import SMCConfig
from proteinsmc.utils.config_unpacker import with_config


class TestWithConfig:
  """Test the with_config decorator."""

  def test_with_config_basic(self, fitness_evaluator_mock: FitnessEvaluator) -> None:
    """Test basic config unpacking."""

    @with_config
    def sample_function(
      sampler_config: BaseSamplerConfig,
      population_size: int,
    ) -> int:
      """Sample function that uses config parameters."""
      return population_size

    config = SMCConfig(
      population_size=1000,
      fitness_evaluator=fitness_evaluator_mock,
    )

    result = sample_function(config)  # type: ignore[call-arg]
    assert result == 1000

  def test_with_config_explicit_override(
    self,
    fitness_evaluator_mock: FitnessEvaluator,
  ) -> None:
    """Test that explicit kwargs override config values."""

    @with_config
    def sample_function(
      sampler_config: BaseSamplerConfig,
      population_size: int,
    ) -> int:
      """Sample function that uses config parameters."""
      return population_size

    config = SMCConfig(
      population_size=1000,
      fitness_evaluator=fitness_evaluator_mock,
    )

    # Override population_size
    result = sample_function(config, population_size=500)
    assert result == 500

  def test_with_config_partial_match(
    self,
    fitness_evaluator_mock: FitnessEvaluator,
  ) -> None:
    """Test config unpacking with partial parameter match."""

    @with_config
    def sample_function(
      sampler_config: BaseSamplerConfig,
      population_size: int,
      extra_param: str = "default",
    ) -> tuple[int, str]:
      """Sample function with extra parameter not in config."""
      return population_size, extra_param

    config = SMCConfig(
      population_size=1000,
      fitness_evaluator=fitness_evaluator_mock,
    )

    result = sample_function(config)  # type: ignore[call-arg]
    assert result == (1000, "default")

  def test_with_config_with_extra_kwargs(
    self,
    fitness_evaluator_mock: FitnessEvaluator,
  ) -> None:
    """Test config unpacking with additional explicit kwargs."""

    @with_config
    def sample_function(
      sampler_config: BaseSamplerConfig,
      population_size: int,
      extra_param: str = "default",
    ) -> tuple[int, str]:
      """Sample function with extra parameter."""
      return population_size, extra_param

    config = SMCConfig(
      population_size=1000,
      fitness_evaluator=fitness_evaluator_mock,
    )

    result = sample_function(config, extra_param="custom")  # type: ignore[call-arg]
    assert result == (1000, "custom")

  def test_with_config_no_matching_params(
    self,
    fitness_evaluator_mock: FitnessEvaluator,
  ) -> None:
    """Test config unpacking when function has no matching parameters."""

    @with_config
    def sample_function(
      sampler_config: BaseSamplerConfig,
      custom_param: str = "value",
    ) -> str:
      """Sample function with no config-matching parameters."""
      return custom_param

    config = SMCConfig(
      population_size=1000,
      fitness_evaluator=fitness_evaluator_mock,
    )

    result = sample_function(config)  # type: ignore[call-arg]
    assert result == "value"

  def test_with_config_preserves_function_metadata(self) -> None:
    """Test that decorator preserves function name and docstring."""

    @with_config
    def sample_function(
      sampler_config: BaseSamplerConfig,
      population_size: int,
    ) -> int:
      """Sample function docstring."""
      return population_size

    assert sample_function.__name__ == "sample_function"
    assert sample_function.__doc__ and "Sample function docstring" in sample_function.__doc__

  def test_with_config_multiple_calls(
    self,
    fitness_evaluator_mock: FitnessEvaluator,
  ) -> None:
    """Test that decorated function can be called multiple times with different configs."""

    @with_config
    def sample_function(
      sampler_config: BaseSamplerConfig,
      population_size: int,
    ) -> int:
      """Sample function."""
      return population_size

    config1 = SMCConfig(population_size=500, fitness_evaluator=fitness_evaluator_mock)
    config2 = SMCConfig(population_size=1000, fitness_evaluator=fitness_evaluator_mock)

    result1 = sample_function(config1)  # type: ignore[call-arg]
    result2 = sample_function(config2)  # type: ignore[call-arg]

    assert result1 == 500
    assert result2 == 1000

  def test_with_config_nested_attributes(
    self,
    fitness_evaluator_mock: FitnessEvaluator,
  ) -> None:
    """Test config unpacking with attributes at different levels."""

    @with_config
    def sample_function(
      sampler_config: BaseSamplerConfig,
      population_size: int,
      mutation_rate: float,
    ) -> tuple[int, float]:
      """Sample function using multiple config attributes."""
      return population_size, mutation_rate

    config = SMCConfig(
      population_size=1000,
      mutation_rate=0.15,
      fitness_evaluator=fitness_evaluator_mock,
    )

    result = sample_function(config)  # type: ignore[call-arg]
    assert result == (1000, 0.15)

  def test_with_config_mixed_override(
    self,
    fitness_evaluator_mock: FitnessEvaluator,
  ) -> None:
    """Test config unpacking with mixed override scenarios."""

    @with_config
    def sample_function(
      sampler_config: BaseSamplerConfig,
      population_size: int,
      mutation_rate: float,
      extra: str = "default",
    ) -> tuple[int, float, str]:
      """Sample function with mixed parameters."""
      return population_size, mutation_rate, extra

    config = SMCConfig(
      population_size=1000,
      mutation_rate=0.1,
      fitness_evaluator=fitness_evaluator_mock,
    )

    # Override one config param and one extra param
    result = sample_function(config, mutation_rate=0.2, extra="custom")  # type: ignore[call-arg]
    assert result == (1000, 0.2, "custom")

  def test_with_config_type_preservation(
    self,
    fitness_evaluator_mock: FitnessEvaluator,
  ) -> None:
    """Test that config unpacking preserves parameter types."""

    @with_config
    def sample_function(
      sampler_config: BaseSamplerConfig,
      population_size: int,
      mutation_rate: float,
    ) -> bool:
      """Sample function that checks types."""
      return isinstance(population_size, int) and isinstance(mutation_rate, float)

    config = SMCConfig(
      population_size=1000,
      mutation_rate=0.1,
      fitness_evaluator=fitness_evaluator_mock,
    )

    assert sample_function(config)  # type: ignore[call-arg]


