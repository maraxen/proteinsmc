"""Data structures for SMC sampling algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp

from proteinsmc.utils import AutoTuningConfig

if TYPE_CHECKING:
  from jaxtyping import Float, Int, PRNGKeyArray

  from proteinsmc.utils import (
    AnnealingScheduleConfig,
    FitnessEvaluator,
    PerGenerationFloat,
    PopulationSequences,
  )


@dataclass(frozen=True)
class MemoryConfig:
  """Configuration for memory efficiency and performance optimization."""

  population_chunk_size: int = 64
  enable_chunked_vmap: bool = True
  device_memory_fraction: float = 0.8
  auto_tuning_config: AutoTuningConfig = field(default_factory=lambda: AutoTuningConfig())

  def _validate_types(self) -> None:
    """Validate the types of the fields."""
    if not isinstance(self.population_chunk_size, int):
      msg = "population_chunk_size must be an integer."
      raise TypeError(msg)
    if not isinstance(self.enable_chunked_vmap, bool):
      msg = "enable_chunked_vmap must be a boolean."
      raise TypeError(msg)
    if not isinstance(self.device_memory_fraction, float):
      msg = "device_memory_fraction must be a float."
      raise TypeError(msg)
    if not isinstance(self.auto_tuning_config, AutoTuningConfig):
      msg = "auto_tuning_config must be an AutoTuningConfig instance."
      raise TypeError(msg)

  def __post_init__(self) -> None:
    """Validate the memory configuration."""
    self._validate_types()
    if not (0.0 < self.device_memory_fraction <= 1.0):
      msg = "device_memory_fraction must be in (0.0, 1.0]."
      raise ValueError(msg)
    if self.population_chunk_size <= 0:
      msg = "population_chunk_size must be positive."
      raise ValueError(msg)

  def tree_flatten(self: MemoryConfig) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children: tuple = ()
    aux_data: dict = dict(self.__dict__.items())
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls: type[MemoryConfig], aux_data: dict, _: tuple) -> MemoryConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(**aux_data)


@dataclass(frozen=True)
class SMCConfig:
  """Configuration for the SMC sampler."""

  template_sequence: str
  population_size: int
  n_states: int
  generations: int
  mutation_rate: float
  diversification_ratio: float
  sequence_type: Literal["protein", "nucleotide"]
  annealing_schedule_config: AnnealingScheduleConfig
  fitness_evaluator: FitnessEvaluator
  memory_config: MemoryConfig = field(default_factory=MemoryConfig)

  def _validate_types(self) -> None:
    """Validate the types of the fields."""
    if not isinstance(self.template_sequence, str):
      msg = "template_sequence must be a string."
      raise TypeError(msg)
    if not isinstance(self.population_size, int):
      msg = "population_size must be an integer."
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

  def __post_init__(self) -> None:
    """Validate the SMC configuration."""
    if self.population_size <= 0:
      msg = "population_size must be positive."
      raise ValueError(msg)
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

  def tree_flatten(self: SMCConfig) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children: tuple = ()
    aux_data: dict = dict(self.__dict__.items())
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls: type[SMCConfig], aux_data: dict, _: tuple) -> SMCConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(**aux_data)


@dataclass
class SMCCarryState:
  """State of the SMC sampler at a single step. Designed to be a JAX PyTree."""

  key: PRNGKeyArray
  population: PopulationSequences
  logZ_estimate: Float  # noqa: N815
  beta: Float
  step: Int = field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))

  def tree_flatten(self: SMCCarryState) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children: tuple = (
      self.key,
      self.population,
      self.logZ_estimate,
      self.beta,
      self.step,
    )
    aux_data: dict = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(
    cls: type[SMCCarryState],
    _aux_data: dict,
    children: tuple,
  ) -> SMCCarryState:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(*children)


@dataclass
class SMCOutput:
  """Output of the SMC sampler."""

  input_config: SMCConfig
  mean_combined_fitness_per_gen: PerGenerationFloat
  max_combined_fitness_per_gen: PerGenerationFloat
  entropy_per_gen: PerGenerationFloat
  beta_per_gen: PerGenerationFloat
  ess_per_gen: PerGenerationFloat
  fitness_components_per_gen: PerGenerationFloat
  final_logZhat: Float  # noqa: N815
  final_amino_acid_entropy: Float

  def tree_flatten(self: SMCOutput) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children: tuple = (
      self.input_config,
      self.mean_combined_fitness_per_gen,
      self.max_combined_fitness_per_gen,
      self.entropy_per_gen,
      self.beta_per_gen,
      self.ess_per_gen,
      self.fitness_components_per_gen,
    )
    aux_data: dict = {
      "final_logZhat": self.final_logZhat,
      "final_amino_acid_entropy": self.final_amino_acid_entropy,
    }
    return (children, aux_data)

  @classmethod
  def tree_unflatten(
    cls: type[SMCOutput],
    aux_data: dict,
    children: tuple,
  ) -> SMCOutput:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(
      input_config=children[0],
      mean_combined_fitness_per_gen=children[1],
      max_combined_fitness_per_gen=children[2],
      entropy_per_gen=children[3],
      beta_per_gen=children[4],
      ess_per_gen=children[5],
      fitness_components_per_gen=children[6],
      **aux_data,
    )


jax.tree_util.register_pytree_node_class(SMCCarryState)
jax.tree_util.register_pytree_node_class(SMCOutput)
