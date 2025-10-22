"""Advanced tests for sampler_base configuration and validation."""

from __future__ import annotations

import jax
import pytest

from proteinsmc.models.annealing import AnnealingConfig
from proteinsmc.models.fitness import CombineFunction, FitnessEvaluator, FitnessFunction
from proteinsmc.models.memory import MemoryConfig
from proteinsmc.models.sampler_base import BaseSamplerConfig


class TestDeviceMeshInitialization:
  """Test device mesh initialization logic."""

  def test_autodetect_mesh(self, fitness_evaluator_mock) -> None:
    """Test automatic device mesh detection.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
    
    Returns:
        None
    
    Raises:
        AssertionError: If mesh autodetection fails.
    
    Example:
        >>> test_autodetect_mesh(fitness_evaluator_mock)
    
    """
    config = BaseSamplerConfig(
      fitness_evaluator=fitness_evaluator_mock,
      seed_sequence="ACDEF",
      # device_mesh_shape=None triggers autodetect
    )

    # Mesh should be created
    assert config.mesh is not None
    # Should have axis names
    assert config.mesh.axis_names is not None
    # Default is 1D mesh
    assert len(config.mesh.axis_names) == 1

  def test_explicit_mesh_shape(self, fitness_evaluator_mock) -> None:
    """Test explicit device mesh shape specification.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
    
    Returns:
        None
    
    Raises:
        AssertionError: If explicit mesh creation fails.
    
    Example:
        >>> test_explicit_mesh_shape(fitness_evaluator_mock)
    
    """
    n_devices = len(jax.devices())
    
    config = BaseSamplerConfig(
      fitness_evaluator=fitness_evaluator_mock,
      seed_sequence="ACDEF",
      device_mesh_shape=(n_devices,),
      axis_names=("devices",),
    )

    assert config.mesh is not None
    assert config.mesh.axis_names == ("devices",)

  def test_mesh_shape_mismatch(self, fitness_evaluator_mock) -> None:
    """Test error when mesh shape doesn't match available devices.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
    
    Returns:
        None
    
    Raises:
        ValueError: If mesh shape doesn't match devices.
    
    Example:
        >>> test_mesh_shape_mismatch(fitness_evaluator_mock)
    
    """
    n_devices = len(jax.devices())
    
    with pytest.raises(ValueError, match="requires.*devices"):
      BaseSamplerConfig(
        fitness_evaluator=fitness_evaluator_mock,
        seed_sequence="ACDEF",
        device_mesh_shape=(n_devices + 1,),
        axis_names=("devices",),
      )

  def test_missing_axis_names(self, fitness_evaluator_mock) -> None:
    """Test error when axis_names not provided with mesh shape.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
    
    Returns:
        None
    
    Raises:
        ValueError: If axis_names not provided.
    
    Example:
        >>> test_missing_axis_names(fitness_evaluator_mock)
    
    """
    n_devices = len(jax.devices())
    
    with pytest.raises(ValueError, match="must provide axis_names"):
      BaseSamplerConfig(
        fitness_evaluator=fitness_evaluator_mock,
        seed_sequence="ACDEF",
        device_mesh_shape=(n_devices,),
        # axis_names missing
      )

  def test_axis_names_length_mismatch(self, fitness_evaluator_mock) -> None:
    """Test error when axis_names length doesn't match mesh shape.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
    
    Returns:
        None
    
    Raises:
        ValueError: If axis_names length doesn't match.
    
    Example:
        >>> test_axis_names_length_mismatch(fitness_evaluator_mock)
    
    """
    n_devices = len(jax.devices())
    
    with pytest.raises(ValueError, match="Length of axis_names.*must match"):
      BaseSamplerConfig(
        fitness_evaluator=fitness_evaluator_mock,
        seed_sequence="ACDEF",
        device_mesh_shape=(n_devices,),
        axis_names=("devices", "extra"),  # Too many names
      )


class TestTypeValidation:
  """Test type validation in BaseSamplerConfig."""

  def test_invalid_fitness_evaluator_type(self) -> None:
    """Test error when fitness_evaluator is wrong type.
    
    Args:
        None
    
    Returns:
        None
    
    Raises:
        TypeError: If fitness_evaluator is wrong type.
    
    Example:
        >>> test_invalid_fitness_evaluator_type()
    
    """
    with pytest.raises(TypeError, match="fitness_evaluator must be"):
      BaseSamplerConfig(
        fitness_evaluator="not_an_evaluator",  # type: ignore[arg-type]
        seed_sequence="ACDEF",
      )

  def test_invalid_seed_sequence_type(self, fitness_evaluator_mock) -> None:
    """Test error when seed_sequence is wrong type.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
    
    Returns:
        None
    
    Raises:
        TypeError: If seed_sequence is wrong type.
    
    Example:
        >>> test_invalid_seed_sequence_type(fitness_evaluator_mock)
    
    """
    with pytest.raises(TypeError, match="seed_sequence must be"):
      BaseSamplerConfig(
        fitness_evaluator=fitness_evaluator_mock,
        seed_sequence=12345,  # type: ignore[arg-type]
      )

  def test_invalid_num_samples_type(self, fitness_evaluator_mock) -> None:
    """Test error when num_samples is wrong type.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
    
    Returns:
        None
    
    Raises:
        TypeError: If num_samples is wrong type.
    
    Example:
        >>> test_invalid_num_samples_type(fitness_evaluator_mock)
    
    """
    with pytest.raises(TypeError, match="num_samples must be"):
      BaseSamplerConfig(
        fitness_evaluator=fitness_evaluator_mock,
        seed_sequence="ACDEF",
        num_samples="not_an_int",  # type: ignore[arg-type]
      )

  def test_invalid_sequence_type(self, fitness_evaluator_mock) -> None:
    """Test error when sequence_type is invalid.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
    
    Returns:
        None
    
    Raises:
        TypeError: If sequence_type is invalid.
    
    Example:
        >>> test_invalid_sequence_type(fitness_evaluator_mock)
    
    """
    with pytest.raises(TypeError, match="sequence_type must be"):
      BaseSamplerConfig(
        fitness_evaluator=fitness_evaluator_mock,
        seed_sequence="ACDEF",
        sequence_type="dna",  # type: ignore[arg-type]
      )

  def test_invalid_memory_config_type(self, fitness_evaluator_mock) -> None:
    """Test error when memory_config is wrong type.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
    
    Returns:
        None
    
    Raises:
        TypeError: If memory_config is wrong type.
    
    Example:
        >>> test_invalid_memory_config_type(fitness_evaluator_mock)
    
    """
    with pytest.raises(TypeError, match="memory_config must be"):
      BaseSamplerConfig(
        fitness_evaluator=fitness_evaluator_mock,
        seed_sequence="ACDEF",
        memory_config="not_a_config",  # type: ignore[arg-type]
      )


class TestValueValidation:
  """Test value validation in BaseSamplerConfig."""

  def test_negative_num_samples(self, fitness_evaluator_mock) -> None:
    """Test error when num_samples is negative.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
    
    Returns:
        None
    
    Raises:
        ValueError: If num_samples is negative.
    
    Example:
        >>> test_negative_num_samples(fitness_evaluator_mock)
    
    """
    with pytest.raises(ValueError, match="num_samples must be positive"):
      BaseSamplerConfig(
        fitness_evaluator=fitness_evaluator_mock,
        seed_sequence="ACDEF",
        num_samples=-10,
      )

  def test_negative_n_states(self, fitness_evaluator_mock) -> None:
    """Test error when n_states is negative.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
    
    Returns:
        None
    
    Raises:
        ValueError: If n_states is negative.
    
    Example:
        >>> test_negative_n_states(fitness_evaluator_mock)
    
    """
    with pytest.raises(ValueError, match="n_states must be positive"):
      BaseSamplerConfig(
        fitness_evaluator=fitness_evaluator_mock,
        seed_sequence="ACDEF",
        n_states=-1,
      )

  def test_mutation_rate_out_of_range(self, fitness_evaluator_mock) -> None:
    """Test error when mutation_rate is out of [0, 1] range.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
    
    Returns:
        None
    
    Raises:
        ValueError: If mutation_rate is out of range.
    
    Example:
        >>> test_mutation_rate_out_of_range(fitness_evaluator_mock)
    
    """
    with pytest.raises(ValueError, match="mutation_rate must be in"):
      BaseSamplerConfig(
        fitness_evaluator=fitness_evaluator_mock,
        seed_sequence="ACDEF",
        mutation_rate=1.5,
      )

  def test_diversification_ratio_out_of_range(self, fitness_evaluator_mock) -> None:
    """Test error when diversification_ratio is out of [0, 1] range.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
    
    Returns:
        None
    
    Raises:
        ValueError: If diversification_ratio is out of range.
    
    Example:
        >>> test_diversification_ratio_out_of_range(fitness_evaluator_mock)
    
    """
    with pytest.raises(ValueError, match="diversification_ratio must be in"):
      BaseSamplerConfig(
        fitness_evaluator=fitness_evaluator_mock,
        seed_sequence="ACDEF",
        diversification_ratio=-0.1,
      )


class TestSequenceListSupport:
  """Test support for sequence lists in config."""

  def test_sequence_list(self, fitness_evaluator_mock) -> None:
    """Test configuration with list of seed sequences.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
    
    Returns:
        None
    
    Raises:
        AssertionError: If sequence list not handled correctly.
    
    Example:
        >>> test_sequence_list(fitness_evaluator_mock)
    
    """
    config = BaseSamplerConfig(
      fitness_evaluator=fitness_evaluator_mock,
      seed_sequence=["ACDEF", "GHIKL"],
    )

    assert isinstance(config.seed_sequence, list)
    assert len(config.seed_sequence) == 2

  def test_num_samples_list(self, fitness_evaluator_mock) -> None:
    """Test configuration with list of num_samples.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
    
    Returns:
        None
    
    Raises:
        AssertionError: If num_samples list not handled correctly.
    
    Example:
        >>> test_num_samples_list(fitness_evaluator_mock)
    
    """
    config = BaseSamplerConfig(
      fitness_evaluator=fitness_evaluator_mock,
      seed_sequence="ACDEF",
      num_samples=[10, 20, 30],
    )

    assert isinstance(config.num_samples, list)
    assert len(config.num_samples) == 3


class TestAnnealingConfig:
  """Test annealing configuration integration."""

  def test_with_annealing_config(self, fitness_evaluator_mock) -> None:
    """Test configuration with annealing schedule.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
    
    Returns:
        None
    
    Raises:
        AssertionError: If annealing config not handled correctly.
    
    Example:
        >>> test_with_annealing_config(fitness_evaluator_mock)
    
    """
    annealing_config = AnnealingConfig(
      annealing_fn="linear",
      n_steps=10,
      beta_max=1.0,
    )

    config = BaseSamplerConfig(
      fitness_evaluator=fitness_evaluator_mock,
      seed_sequence="ACDEF",
      annealing_config=annealing_config,
    )

    assert config.annealing_config is not None
    assert config.annealing_config.annealing_fn == "linear"
    assert config.annealing_config.n_steps == 10

  def test_without_annealing_config(self, fitness_evaluator_mock) -> None:
    """Test configuration without annealing schedule.
    
    Args:
        fitness_evaluator_mock: Fixture providing mock fitness evaluator.
    
    Returns:
        None
    
    Raises:
        AssertionError: If default annealing config is incorrect.
    
    Example:
        >>> test_without_annealing_config(fitness_evaluator_mock)
    
    """
    config = BaseSamplerConfig(
      fitness_evaluator=fitness_evaluator_mock,
      seed_sequence="ACDEF",
    )

    assert config.annealing_config is None
