"""Data structures for SMC sampling algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (  # Added Any, Callable for clarity
  TYPE_CHECKING,
  Any,
  Callable,
  Literal,
  Protocol,
)

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from jaxtyping import Float, Int, PRNGKeyArray


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

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility.

    All fields are treated as children as they can vary across instances.
    """
    children = (
      self.population_chunk_size,
      self.enable_chunked_vmap,
      self.device_memory_fraction,
      self.auto_tuning_config,
    )
    aux_data = {}  # aux_data is empty as all varying fields are children
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data: dict, children: tuple) -> MemoryConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(
      population_chunk_size=children[0],
      enable_chunked_vmap=children[1],
      device_memory_fraction=children[2],
      auto_tuning_config=children[3],
      **aux_data,
    )


@dataclass(frozen=True)
class AutoTuningConfig:
  """Configuration for automatic chunk size tuning."""

  enable_auto_tuning: bool = True
  probe_chunk_sizes: tuple[int, ...] = (16, 32, 64, 128, 256)
  max_probe_iterations: int = 3
  memory_safety_factor: float = 0.8
  performance_tolerance: float = 0.1

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility.

    All fields are treated as children as they can vary across instances.
    """
    children = (
      self.enable_auto_tuning,
      self.probe_chunk_sizes,
      self.max_probe_iterations,
      self.memory_safety_factor,
      self.performance_tolerance,
    )
    aux_data = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data: dict, children: tuple) -> AutoTuningConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(
      enable_auto_tuning=children[0],
      probe_chunk_sizes=children[1],
      max_probe_iterations=children[2],
      memory_safety_factor=children[3],
      performance_tolerance=children[4],
      **aux_data,
    )


@dataclass(frozen=True)
class AnnealingScheduleConfig:
  """Configuration for an annealing schedule.

  Attributes:
      schedule_fn: Callable function that defines the annealing schedule.
      beta_max: Maximum value for beta.
      annealing_len: Number of steps over which to anneal.
      schedule_args: Additional arguments for the schedule function.

  """

  schedule_fn: Callable[[Int, Int, Float], Float]
  beta_max: float
  annealing_len: int
  schedule_args: tuple = field(default_factory=tuple)

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility.

    All fields are treated as children as they can vary across instances.
    """
    children = (
      self.beta_max,
      self.annealing_len,
      self.schedule_args,
    )
    aux_data = {"schedule_fn": self.schedule_fn}
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data: dict, children: tuple) -> AnnealingScheduleConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(
      schedule_fn=aux_data["schedule_fn"],
      beta_max=children[0],
      annealing_len=children[1],
      schedule_args=children[2],
    )


@dataclass(frozen=True)
class FitnessFunction:
  """Represents a single fitness function."""

  func: Callable[[PRNGKeyArray, Int], Float]
  input_type: Literal["protein", "nucleotide"]
  name: str = "unnamed_fitness_function"

  def __post_init__(self) -> None:
    """Validate the fitness function metadata."""
    if not callable(self.func):
      msg = f"Fitness function {self.func} is not callable."
      raise TypeError(msg)
    if self.input_type not in ["nucleotide", "protein"]:
      msg = f"Invalid input_type '{self.input_type}'. Expected 'nucleotide' or 'protein'."
      raise ValueError(
        msg,
      )
    if not isinstance(self.name, str):
      msg = "name must be a string."
      raise TypeError(msg)

  def __hash__(self) -> int:
    """Hash the fitness function based on its properties."""
    return hash((self.func, self.input_type, frozenset(self.name)))

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility.

    All fields are treated as children as they can vary across instances.
    """
    children = (
      self.input_type,
      self.name,
    )
    aux_data = {"func": self.func}
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data: dict, children: tuple) -> FitnessFunction:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(
      func=aux_data["func"],
      input_type=children[0],
      name=children[1],
    )


@dataclass(frozen=True)
class FitnessEvaluator:
  """Manages multiple fitness functions and their combination."""

  fitness_functions: tuple[FitnessFunction, ...]
  combine_func: Callable[[Float], Float] | None = None

  def __post_init__(self) -> None:
    """Validate the fitness evaluator configuration."""
    if not self.fitness_functions:
      msg = "At least one fitness function must be provided."
      raise ValueError(msg)

  def get_functions_by_type(
    self,
    input_type: Literal["nucleotide", "protein"],
  ) -> list[FitnessFunction]:
    """Get active fitness functions that accept the specified input type."""
    return [f for f in self.fitness_functions if f.input_type == input_type]

  def __hash__(self) -> int:
    """Hash the fitness evaluator based on its properties."""
    return hash(tuple(sorted(self.fitness_functions, key=lambda f: f.name)))

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility.

    All fields are treated as children as they can vary across instances.
    """
    children = (self.fitness_functions,)
    aux_data = {"combine_func": self.combine_func}
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data: dict, children: tuple) -> FitnessEvaluator:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(
      fitness_functions=children[0],
      combine_func=aux_data["combine_func"],
    )


@dataclass(frozen=True)
class BaseSamplerConfig:
  """Base configuration for samplers.

  All sampler configurations should inherit from this.
  """

  seed_sequence: str
  generations: int
  n_states: int
  mutation_rate: float
  diversification_ratio: float
  sequence_type: Literal["protein", "nucleotide"]
  fitness_evaluator: FitnessEvaluator
  memory_config: MemoryConfig

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

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility.

    All fields are treated as children as they can vary across instances.
    """
    children = (
      self.seed_sequence,
      self.generations,
      self.n_states,
      self.mutation_rate,
      self.diversification_ratio,
      self.sequence_type,
      self.fitness_evaluator,
      self.memory_config,
    )
    aux_data = {}  # aux_data is empty as all varying fields are children
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data: dict, children: tuple) -> BaseSamplerConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(
      seed_sequence=children[0],
      generations=children[1],
      n_states=children[2],
      mutation_rate=children[3],
      diversification_ratio=children[4],
      sequence_type=children[5],
      fitness_evaluator=children[6],
      memory_config=children[7],
      **aux_data,
    )

  @property
  def additional_config_fields(self) -> dict[str, str]:
    """Return additional fields for the configuration that are not part of the PyTree."""
    return {}


class SamplerOutputProtocol(Protocol):
  """Protocol for sampler output objects, for generic data extraction."""

  @property
  def input_configs(self) -> BaseSamplerConfig | tuple[BaseSamplerConfig, ...]:
    """Return the input configuration(s) for the sampler output.

    This can be a single config or a tuple of configs if batched.
    """
    ...

  @property
  def per_gen_stats_metrics(self) -> dict[str, str]:
    """Return a mapping from generic metric name to attribute name for per-generation stats."""
    ...

  @property
  def summary_stats_metrics(self) -> dict[str, str]:
    """Return a mapping from generic metric name to attribute name for summary stats."""
    ...

  @property
  def output_type_name(self) -> str:
    """Return the string name of the output type (e.g., 'SMC', 'ParallelReplicaSMC')."""
    ...


@dataclass(frozen=True)
class SMCConfig(BaseSamplerConfig):
  """Configuration for the SMC sampler."""

  population_size: int
  annealing_schedule: AnnealingScheduleConfig

  def _validate_types(self) -> None:
    """Validate the types of the fields."""
    if not isinstance(self.population_size, int):
      msg = "population_size must be an integer."
      raise TypeError(msg)
    if not isinstance(self.annealing_schedule, AnnealingScheduleConfig):
      msg = "annealing_schedule must be an AnnealingScheduleConfig instance."
      raise TypeError(msg)
    super()._validate_types()

  def __post_init__(self: SMCConfig) -> None:
    """Validate the SMC configuration."""
    self._validate_types()
    if self.population_size <= 0:
      msg = "population_size must be positive."
      raise ValueError(msg)
    super().__post_init__()

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility.

    All fields are treated as children as they can vary across instances.
    """
    children = (
      self.seed_sequence,
      self.generations,
      self.n_states,
      self.mutation_rate,
      self.diversification_ratio,
      self.sequence_type,
      self.fitness_evaluator,
      self.memory_config,
      self.population_size,
      self.annealing_schedule,
    )
    aux_data = {}  # aux_data is empty as all varying fields are children
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data: dict, children: tuple) -> SMCConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(
      seed_sequence=children[0],
      generations=children[1],
      n_states=children[2],
      mutation_rate=children[3],
      diversification_ratio=children[4],
      sequence_type=children[5],
      fitness_evaluator=children[6],
      memory_config=children[7],
      population_size=children[8],
      annealing_schedule=children[9],
      **aux_data,
    )

  @property
  def additional_config_fields(self) -> dict[str, str]:
    """Return additional fields for the configuration that are not part of the PyTree."""
    return {
      "population_size": "population_size",
      "annealing_schedule": "annealing_schedule",
    }


@dataclass
class SMCCarryState:
  """State of the SMC sampler at a single step. Designed to be a JAX PyTree."""

  key: PRNGKeyArray
  population: Any  # Changed from PopulationSequences to Any for broader compatibility
  logZ_estimate: Float  # noqa: N815
  beta: Float
  step: Int = field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))

  def tree_flatten(self) -> tuple[tuple, dict]:
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
    cls,
    _aux_data: dict,
    children: tuple,
  ) -> SMCCarryState:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(*children)


@dataclass
class SMCOutput(SamplerOutputProtocol):
  """Output of the SMC sampler."""

  input_config: SMCConfig
  mean_combined_fitness_per_gen: Float
  max_combined_fitness_per_gen: Float
  entropy_per_gen: Float
  beta_per_gen: Float
  ess_per_gen: Float
  fitness_components_per_gen: Float
  final_logZhat: Float  # noqa: N815
  final_amino_acid_entropy: Float

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children: tuple = (
      self.input_config,
      self.mean_combined_fitness_per_gen,
      self.max_combined_fitness_per_gen,
      self.entropy_per_gen,
      self.beta_per_gen,
      self.ess_per_gen,
      self.fitness_components_per_gen,
      self.final_logZhat,
      self.final_amino_acid_entropy,
    )
    aux_data: dict = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(
    cls,
    _aux_data: dict,
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
      final_logZhat=children[7],
      final_amino_acid_entropy=children[8],
    )

  @property
  def input_configs(self) -> SMCConfig | tuple[SMCConfig, ...]:
    """Return the input configuration(s) for the experiment output.

    This can be a single config or a tuple of configs if batched.
    """
    return self.input_config

  @property
  def per_gen_stats_metrics(self) -> dict[str, str]:
    """Return mappings for SMC per-generation metrics."""
    return {
      "mean_fitness": "mean_combined_fitness_per_gen",
      "max_fitness": "max_combined_fitness_per_gen",
      "entropy": "entropy_per_gen",
      "beta": "beta_per_gen",
      "ess": "ess_per_gen",
      "fitness_components": "fitness_components_per_gen",
    }

  @property
  def summary_stats_metrics(self) -> dict[str, str]:
    """Return mappings for SMC summary metrics."""
    return {
      "final_logZhat": "final_logZhat",
      "final_amino_acid_entropy": "final_amino_acid_entropy",
    }

  @property
  def output_type_name(self) -> str:
    """Return the string name of the output type."""
    return "SMC"


@dataclass(frozen=True)
class ExchangeConfig:
  """Configuration for parallel replica exchange."""

  population_size_per_island: int
  n_islands: int
  n_exchange_attempts: int
  fitness_evaluator: FitnessEvaluator
  exchange_frequency: float
  sequence_type: Literal["protein", "nucleotide"]
  n_exchange_attempts_per_cycle: int
  ess_threshold_fraction: float

  def _validate_types(self) -> None:
    """Validate the types of the fields."""
    if not isinstance(self.population_size_per_island, int):
      msg = "population_size_per_island must be an integer."
      raise TypeError(msg)
    if not isinstance(self.n_islands, int):
      msg = "n_islands must be an integer."
      raise TypeError(msg)
    if not isinstance(self.n_exchange_attempts, int):
      msg = "n_exchange_attempts must be an integer."
      raise TypeError(msg)
    if not isinstance(self.fitness_evaluator, FitnessEvaluator):
      msg = "fitness_evaluator must be a FitnessEvaluator instance."
      raise TypeError(msg)
    if not isinstance(self.exchange_frequency, float):
      msg = "exchange_frequency must be a float."
      raise TypeError(msg)
    if self.sequence_type not in ("protein", "nucleotide"):
      msg = "sequence_type must be 'protein' or 'nucleotide'."
      raise ValueError(msg)
    if not isinstance(self.n_exchange_attempts_per_cycle, int):
      msg = "n_exchange_attempts_per_cycle must be an integer."
      raise TypeError(msg)
    if not isinstance(self.ess_threshold_fraction, float):
      msg = "ess_threshold_fraction must be a float."
      raise TypeError(msg)

  def __post_init__(self: ExchangeConfig) -> None:
    """Validate the exchange configuration."""
    self._validate_types()
    if self.population_size_per_island <= 0:
      msg = "population_size_per_island must be positive."
      raise ValueError(msg)
    if self.n_islands <= 0:
      msg = "n_islands must be positive."
      raise ValueError(msg)
    if self.n_exchange_attempts <= 0:
      msg = "n_exchange_attempts must be positive."
      raise ValueError(msg)
    if self.exchange_frequency < 0.0 or self.exchange_frequency > 1.0:
      msg = "exchange_frequency must be in [0.0, 1.0]."
      raise ValueError(msg)

  def tree_flatten(self: ExchangeConfig) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility.

    All fields are treated as children as they can vary across instances.
    """
    children = (
      self.population_size_per_island,
      self.n_islands,
      self.n_exchange_attempts,
      self.fitness_evaluator,
      self.exchange_frequency,
      self.sequence_type,
      self.n_exchange_attempts_per_cycle,
      self.ess_threshold_fraction,
    )
    aux_data = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(
    cls: type[ExchangeConfig],
    aux_data: dict,
    children: tuple,
  ) -> ExchangeConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(
      population_size_per_island=children[0],
      n_islands=children[1],
      n_exchange_attempts=children[2],
      fitness_evaluator=children[3],
      exchange_frequency=children[4],
      sequence_type=children[5],
      n_exchange_attempts_per_cycle=children[6],
      ess_threshold_fraction=children[7],
      **aux_data,
    )


@dataclass(frozen=True)
class PRSMCStepConfig:
  """Configuration for a single step in the PRSMC algorithm."""

  population_size_per_island: int
  mutation_rate: float
  fitness_evaluator: FitnessEvaluator
  sequence_type: Literal["protein", "nucleotide"]
  ess_threshold_frac: float
  meta_beta_annealing_schedule: AnnealingScheduleConfig
  exchange_config: ExchangeConfig

  def _validate_types(self) -> None:
    """Validate the types of the fields."""
    if not isinstance(self.population_size_per_island, int):
      msg = "population_size_per_island must be an integer."
      raise TypeError(msg)
    if not isinstance(self.mutation_rate, float):
      msg = "mutation_rate must be a float."
      raise TypeError(msg)
    if not isinstance(self.fitness_evaluator, FitnessEvaluator):
      msg = "fitness_evaluator must be a FitnessEvaluator instance."
      raise TypeError(msg)
    if self.sequence_type not in ("protein", "nucleotide"):
      msg = "sequence_type must be 'protein' or 'nucleotide'."
      raise ValueError(msg)
    if not isinstance(self.ess_threshold_frac, float):
      msg = "ess_threshold_frac must be a float."
      raise TypeError(msg)
    if not isinstance(self.meta_beta_annealing_schedule, AnnealingScheduleConfig):
      msg = "meta_beta_annealing_schedule must be an AnnealingScheduleConfig instance."
      raise TypeError(msg)
    if not isinstance(self.exchange_config, ExchangeConfig):
      msg = "exchange_config must be an ExchangeConfig instance."
      raise TypeError(msg)

  def __post_init__(self: PRSMCStepConfig) -> None:
    """Validate the PRSMC step configuration."""
    self._validate_types()
    if self.population_size_per_island <= 0:
      msg = "population_size_per_island must be positive."
      raise ValueError(msg)
    if not (0.0 <= self.mutation_rate <= 1.0):
      msg = "mutation_rate must be in [0.0, 1.0]."
      raise ValueError(msg)
    if not (0.0 < self.ess_threshold_frac <= 1.0):
      msg = "ess_threshold_frac must be in (0.0, 1.0]."
      raise ValueError(msg)
    if self.sequence_type not in ("protein", "nucleotide"):
      msg = "sequence_type must be 'protein' or 'nucleotide'."
      raise ValueError(msg)

  def tree_flatten(self: PRSMCStepConfig) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility.

    All fields are treated as children as they can vary across instances.
    """
    children = (
      self.population_size_per_island,
      self.mutation_rate,
      self.fitness_evaluator,
      self.sequence_type,
      self.ess_threshold_frac,
      self.meta_beta_annealing_schedule,
      self.exchange_config,
    )
    aux_data = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(
    cls: type[PRSMCStepConfig],
    aux_data: dict,
    children: tuple,
  ) -> PRSMCStepConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(
      population_size_per_island=children[0],
      mutation_rate=children[1],
      fitness_evaluator=children[2],
      sequence_type=children[3],
      ess_threshold_frac=children[4],
      meta_beta_annealing_schedule=children[5],
      exchange_config=children[6],
      **aux_data,
    )


@dataclass(frozen=True)
class ParallelReplicaConfig(BaseSamplerConfig):
  """Configuration for parallel replica SMC simulation."""

  population_size_per_island: int
  n_islands: int
  island_betas: list[float]
  step_config: PRSMCStepConfig

  def _validate_types(self) -> None:
    """Validate the types of the fields."""
    super()._validate_types()
    if not isinstance(self.n_islands, int):
      msg = "n_islands must be an integer."
      raise TypeError(msg)
    if not isinstance(self.island_betas, list):
      msg = "island_betas must be a list of floats."
      raise TypeError(msg)
    if not all(isinstance(beta, float) for beta in self.island_betas):
      msg = "island_betas must be a list of floats."
      raise TypeError(msg)
    if not isinstance(self.step_config, PRSMCStepConfig):
      msg = "step_config must be a PRSMCStepConfig instance."
      raise TypeError(msg)

  def __post_init__(self: ParallelReplicaConfig) -> None:
    """Validate the SMC configuration."""
    self._validate_types()
    super().__post_init__()
    if self.n_islands <= 0:
      msg = "n_islands must be positive."
      raise ValueError(msg)
    if len(self.island_betas) != self.n_islands:
      msg = (
        f"Length of island_betas ({len(self.island_betas)}) must match "
        f"n_islands ({self.n_islands})."
      )
      raise ValueError(msg)
    if not all(0.0 <= beta <= 1.0 for beta in self.island_betas):
      msg = "All island_betas must be in [0.0, 1.0]."
      raise ValueError(msg)
    if not isinstance(self.step_config, PRSMCStepConfig):
      msg = "step_config must be a PRSMCStepConfig instance."
      raise TypeError(msg)

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility.

    All fields are treated as children as they can vary across instances.
    """
    children = (
      self.seed_sequence,
      self.generations,
      self.n_states,
      self.mutation_rate,
      self.diversification_ratio,
      self.sequence_type,
      self.fitness_evaluator,
      self.memory_config,
      self.n_islands,
      self.island_betas,
      self.step_config,
    )
    aux_data = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data: dict, children: tuple) -> ParallelReplicaConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(
      seed_sequence=children[0],
      generations=children[1],
      n_states=children[2],
      mutation_rate=children[3],
      diversification_ratio=children[4],
      sequence_type=children[5],
      fitness_evaluator=children[6],
      memory_config=children[7],
      n_islands=children[8],
      island_betas=list(children[9]),  # Convert tuple back to list
      step_config=children[10],
      **aux_data,
    )

  @property
  def additional_config_fields(self) -> dict[str, str]:
    """Return additional fields for the configuration that are not part of the PyTree."""
    return {
      "n_islands": "n_islands",
      "island_betas": "island_betas",
      "island_config": "step_config",
    }


@dataclass
class IslandState:
  """State of a single island in the PRSMC algorithm."""

  key: PRNGKeyArray
  population: Int
  beta: Float
  logZ_estimate: Float  # noqa: N815
  ess: Float
  mean_fitness: Float
  step: Int = field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))

  def tree_flatten(self: IslandState) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children: tuple = (
      self.key,
      self.population,
      self.beta,
      self.logZ_estimate,
      self.ess,
      self.mean_fitness,
      self.step,
    )
    aux_data: dict = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(cls: type[IslandState], _aux_data: dict, children: tuple) -> IslandState:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(*children)


@dataclass
class PRSMCCarryState:
  """Carry state for the PRSMC algorithm, containing overall state and PRNG key."""

  current_overall_state: IslandState
  prng_key: PRNGKeyArray
  total_swaps_attempted: Int = field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))
  total_swaps_accepted: Int = field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))

  def tree_flatten(self: PRSMCCarryState) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children: tuple = (
      self.current_overall_state,
      self.prng_key,
      self.total_swaps_attempted,
      self.total_swaps_accepted,
    )
    aux_data: dict = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(
    cls: type[PRSMCCarryState],
    _aux_data: dict,
    children: tuple,
  ) -> PRSMCCarryState:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(*children)


@dataclass
class ParallelReplicaSMCOutput(SamplerOutputProtocol):
  """Output of the Parallel Replica SMC algorithm."""

  input_config: ParallelReplicaConfig
  final_island_states: IslandState
  swap_acceptance_rate: Float
  history_mean_fitness_per_island: Float
  history_max_fitness_per_island: Float
  history_ess_per_island: Float
  history_logZ_increment_per_island: Float  # noqa: N815
  history_meta_beta: Float
  history_num_accepted_swaps: Float
  history_num_attempted_swaps: Float

  def tree_flatten(self: ParallelReplicaSMCOutput) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children: tuple = (
      self.input_config,
      self.final_island_states,
      self.swap_acceptance_rate,
      self.history_mean_fitness_per_island,
      self.history_max_fitness_per_island,
      self.history_ess_per_island,
      self.history_logZ_increment_per_island,
      self.history_meta_beta,
      self.history_num_accepted_swaps,
      self.history_num_attempted_swaps,
    )
    aux_data: dict = {}
    return children, aux_data

  @classmethod
  def tree_unflatten(
    cls: type[ParallelReplicaSMCOutput],
    _aux_data: dict,
    children: tuple,
  ) -> ParallelReplicaSMCOutput:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(
      input_config=children[0],
      final_island_states=children[1],
      swap_acceptance_rate=children[2],
      history_mean_fitness_per_island=children[3],
      history_max_fitness_per_island=children[4],
      history_ess_per_island=children[5],
      history_logZ_increment_per_island=children[6],
      history_meta_beta=children[7],
      history_num_accepted_swaps=children[8],
      history_num_attempted_swaps=children[9],
    )

  @property
  def input_configs(self) -> ParallelReplicaConfig:
    """Return the input configuration(s) for the experiment output."""
    return self.input_config

  @property
  def per_gen_stats_metrics(self) -> dict[str, str]:
    """Return mappings for Parallel Replica SMC per-generation metrics."""
    return {
      "mean_fitness": "history_mean_fitness_per_island",
      "max_fitness": "history_max_fitness_per_island",
      "ess": "history_ess_per_island",
      "logZ_increment": "history_logZ_increment_per_island",
      "meta_beta": "history_meta_beta",
      "num_accepted_swaps": "history_num_accepted_swaps",
      "num_attempted_swaps": "history_num_attempted_swaps",
    }

  @property
  def summary_stats_metrics(self) -> dict[str, str]:
    """Return mappings for Parallel Replica SMC summary metrics."""
    return {
      "final_island_states": "final_island_states",
      "swap_acceptance_rate": "swap_acceptance_rate",
    }

  @property
  def output_type_name(self) -> str:
    """Return the string name of the output type."""
    return "ParallelReplicaSMC"


jax.tree_util.register_pytree_node_class(MemoryConfig)
jax.tree_util.register_pytree_node_class(AutoTuningConfig)
jax.tree_util.register_pytree_node_class(AnnealingScheduleConfig)
jax.tree_util.register_pytree_node_class(FitnessFunction)
jax.tree_util.register_pytree_node_class(FitnessEvaluator)
jax.tree_util.register_pytree_node_class(BaseSamplerConfig)
jax.tree_util.register_pytree_node_class(SMCConfig)
jax.tree_util.register_pytree_node_class(SMCCarryState)
jax.tree_util.register_pytree_node_class(SMCOutput)
jax.tree_util.register_pytree_node_class(ExchangeConfig)
jax.tree_util.register_pytree_node_class(PRSMCStepConfig)
jax.tree_util.register_pytree_node_class(ParallelReplicaConfig)
jax.tree_util.register_pytree_node_class(IslandState)
jax.tree_util.register_pytree_node_class(PRSMCCarryState)
jax.tree_util.register_pytree_node_class(ParallelReplicaSMCOutput)
