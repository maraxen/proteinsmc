"""Base configuration and protocol definitions for samplers in the proteinsmc package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from proteinsmc.models.annealing import AnnealingConfig
from proteinsmc.models.fitness import FitnessEvaluator
from proteinsmc.models.memory import MemoryConfig


@dataclass(frozen=True)
class BaseSamplerConfig:
  """Base configuration for samplers.

  All sampler configurations should inherit from this.
  """

  sampler_type: str
  seed_sequence: str
  generations: int
  n_states: int
  mutation_rate: float
  diversification_ratio: float
  sequence_type: Literal["protein", "nucleotide"]
  fitness_evaluator: FitnessEvaluator
  memory_config: MemoryConfig
  annealing_config: AnnealingConfig

  def _validate_types(self) -> None:
    """Validate the types of the fields."""
    if not isinstance(self.seed_sequence, str):
      msg = "seed_sequence must be a string."
      raise TypeError(msg)
    if not isinstance(self.n_states, int):
      msg = "n_states must be an integer."
      raise TypeError(msg)
    if not isinstance(self.generations, int):
      msg = "generations must be an integer."
      raise TypeError(msg)
    if not isinstance(self.mutation_rate, float):
      msg = "mutation_rate must be a float."
      raise TypeError(msg)
    if not isinstance(self.diversification_ratio, float):
      msg = "diversification_ratio must be a float."
      raise TypeError(msg)
    if not isinstance(self.fitness_evaluator, FitnessEvaluator):
      msg = "fitness_evaluator must be a FitnessEvaluator instance."
      raise TypeError(msg)
    if not isinstance(self.memory_config, MemoryConfig):
      msg = "memory_config must be a MemoryConfig instance."
      raise TypeError(msg)

  def __post_init__(self) -> None:
    """Validate the common configuration fields."""
    if self.n_states <= 0:
      msg = "n_states must be positive."
      raise ValueError(msg)
    if self.generations <= 0:
      msg = "generations must be positive."
      raise ValueError(msg)
    if not (0.0 <= self.mutation_rate <= 1.0):
      msg = "mutation_rate must be in [0.0, 1.0]."
      raise ValueError(msg)
    if not (0.0 <= self.diversification_ratio <= 1.0):
      msg = "diversification_ratio must be in [0.0, 1.0]."
      raise ValueError(msg)
    if self.sequence_type not in ("protein", "nucleotide"):
      msg = "sequence_type must be 'protein' or 'nucleotide'."
      raise ValueError(msg)

  @property
  def additional_config_fields(self) -> dict[str, str]:
    """Return additional fields for the configuration that are not part of the PyTree."""
    return {}


class SamplerOutputProtocol(Protocol):
  """Protocol for sampler output objects, for generic data extraction."""

  @property
  def per_gen_stats_metrics(self) -> dict[str, str]:
    """Return a mapping from generic metric name to attribute name for per-generation stats."""
    return {}

  @property
  def summary_stats_metrics(self) -> dict[str, str]:
    """Return a mapping from generic metric name to attribute name for summary stats."""
    return {}

  @property
  def output_type_name(self) -> str:
    """Return the string name of the output type (e.g., 'SMC', 'ParallelReplicaSMC')."""
    return "SamplerOutput"
