"""Base configuration and protocol definitions for samplers in the proteinsmc package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from proteinsmc.models.fitness import FitnessEvaluator
from proteinsmc.models.memory import MemoryConfig

if TYPE_CHECKING:
  from proteinsmc.models.annealing import AnnealingConfig


@dataclass(frozen=True)
class BaseSamplerConfig:
  """Base configuration for samplers.

  All sampler configurations should inherit from this.
  """

  prng_seed: int = field(default=42)
  """Random seed for reproducibility."""
  sampler_type: str = field(default="unknown")
  """Type of the sampler (e.g., 'gibbs', 'smc', etc.)."""
  """This is used to identify the sampler type in the registry."""
  seed_sequence: str = field(default="")
  """Initial sequence to start the sampling process."""
  num_samples: int = field(default=100)
  """Number of generations to run the sampler."""
  """This is used to control the number of iterations in the sampling process."""
  n_states: int = field(default=20)
  """Number of possible states for each position in the sequence."""
  """This is used to define the state space of the sequences."""
  mutation_rate: float = field(default=0.1)
  """Rate of mutation applied to the sequences during sampling."""
  """This is used to control the diversity of the sampled sequences."""
  diversification_ratio: float = field(default=0.0)
  """Ratio of diversification applied to the sequences."""
  sequence_type: Literal["protein", "nucleotide"] = field(default="protein")
  fitness_evaluator: FitnessEvaluator = field(kw_only=True)
  memory_config: MemoryConfig = field(default_factory=MemoryConfig)
  annealing_config: AnnealingConfig = field(kw_only=True)

  def _validate_types(self) -> None:
    """Validate the types of the fields."""
    if not isinstance(self.seed_sequence, str):
      msg = "seed_sequence must be a string."
      raise TypeError(msg)
    if not isinstance(self.n_states, int):
      msg = "n_states must be an integer."
      raise TypeError(msg)
    if not isinstance(self.num_samples, int):
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
    self._validate_types()
    if self.n_states <= 0:
      msg = "n_states must be positive."
      raise ValueError(msg)
    if self.num_samples <= 0:
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

  @property
  def generations(self) -> int:
    """Alias for num_samples for backward compatibility."""
    return self.num_samples


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
