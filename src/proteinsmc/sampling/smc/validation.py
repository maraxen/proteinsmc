"""Configuration validation utilities for SMC sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from proteinsmc.utils.data_structures import SMCConfig


def validate_smc_config(config: SMCConfig) -> None:
  """Validate the SMCConfig object to ensure all required fields are set correctly.

  Args:
    config: SMCConfig object to validate.

  Raises:
    ValueError or TypeError if any validation fails.

  """
  _validate_config_type(config)
  _validate_sequence_fields(config)
  _validate_numeric_fields(config)
  _validate_component_fields(config)
  _validate_memory_config(config)


def _validate_config_type(config: SMCConfig) -> None:
  """Validate the config type."""
  if not hasattr(config, "__class__") or config.__class__.__name__ != "SMCConfig":
    msg = f"Expected config to be an instance of SMCConfig, got {type(config)}"
    raise TypeError(msg)


def _validate_sequence_fields(config: SMCConfig) -> None:
  """Validate sequence-related fields."""
  if config.seed_sequence is None or config.seed_sequence == "":
    msg = "Template sequence must be provided and cannot be empty."
    raise ValueError(msg)

  if config.sequence_type not in ["protein", "nucleotide"]:
    msg = f"Invalid sequence type '{config.sequence_type}'. Must be 'protein' or 'nucleotide'."
    raise ValueError(msg)


def _validate_numeric_fields(config: SMCConfig) -> None:
  """Validate numeric fields."""
  if config.n_states <= 0:
    msg = f"Number of states must be positive, got {config.n_states}."
    raise ValueError(msg)

  if config.generations <= 0:
    msg = f"Number of generations must be positive, got {config.generations}."
    raise ValueError(msg)

  if config.mutation_rate < 0 or config.mutation_rate > 1:
    msg = f"Mutation rate must be in the range [0, 1], got {config.mutation_rate}."
    raise ValueError(msg)

  if not (0 <= config.diversification_ratio <= 1):
    msg = f"Diversification ratio must be in the range [0, 1], got {config.diversification_ratio}."
    raise ValueError(msg)


def _validate_component_fields(config: SMCConfig) -> None:
  """Validate component fields (fitness evaluator, annealing schedule)."""
  from proteinsmc.utils.annealing_schedules import AnnealingScheduleConfig
  from proteinsmc.utils.fitness import FitnessEvaluator

  if not isinstance(config.fitness_evaluator, FitnessEvaluator):
    msg = (
      f"Expected fitness_evaluator to be an instance of FitnessEvaluator, "
      f"got {type(config.fitness_evaluator)}"
    )
    raise TypeError(msg)

  if not isinstance(config.annealing_schedule_config, AnnealingScheduleConfig):
    msg = (
      f"Expected annealing_schedule_config to be an instance of "
      f"AnnealingScheduleConfig, got {type(config.annealing_schedule_config)}"
    )
    raise TypeError(msg)


def _validate_memory_config(config: SMCConfig) -> None:
  """Validate memory configuration."""
  if config.memory_config.population_chunk_size <= 0:
    msg = (
      f"Population chunk size must be positive, "
      f"got {config.memory_config.population_chunk_size}."
    )
    raise ValueError(msg)

  if not (0 < config.memory_config.device_memory_fraction <= 1):
    msg = (
      f"Device memory fraction must be in the range (0, 1], "
      f"got {config.memory_config.device_memory_fraction}."
    )
    raise ValueError(msg)


def validate_population_size_compatibility(
  population_size: int,
  chunk_size: int,
) -> None:
  """Validate that population size and chunk size are compatible.

  Args:
    population_size: Size of the population
    chunk_size: Size of chunks for processing

  Raises:
    ValueError if the sizes are incompatible.

  """
  if population_size <= 0:
    msg = f"Population size must be positive, got {population_size}."
    raise ValueError(msg)

  if chunk_size <= 0:
    msg = f"Chunk size must be positive, got {chunk_size}."
    raise ValueError(msg)

  if chunk_size > population_size:
    msg = f"Chunk size ({chunk_size}) cannot be larger than population size ({population_size})."
    raise ValueError(msg)


def validate_memory_requirements(
  population_size: int,
  sequence_length: int,
  available_memory_mb: float,
) -> None:
  """Validate that memory requirements can be satisfied.

  Args:
    population_size: Size of the population
    sequence_length: Length of sequences
    available_memory_mb: Available memory in MB

  Raises:
    ValueError if memory requirements cannot be satisfied.

  """
  from proteinsmc.utils import estimate_memory_usage

  required_memory = estimate_memory_usage(population_size, sequence_length)

  if required_memory > available_memory_mb:
    msg = (
      f"Estimated memory requirement ({required_memory:.1f} MB) exceeds "
      f"available memory ({available_memory_mb:.1f} MB). Consider reducing "
      f"population size or using smaller chunks."
    )
    raise ValueError(msg)
