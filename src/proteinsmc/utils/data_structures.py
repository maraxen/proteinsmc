"""Data structures for SMC sampling algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol  # Added Any, Callable for clarity

import jax
import jax.numpy as jnp

from proteinsmc.utils import AnnealingScheduleConfig, AutoTuningConfig, FitnessEvaluator

if TYPE_CHECKING:
  from jaxtyping import Float, Int, PRNGKeyArray

  from proteinsmc.utils import (
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

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children: tuple = ()
    aux_data: dict = dict(self.__dict__.items())
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data: dict, _: tuple) -> MemoryConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(**aux_data)


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
    """Flatten the dataclass for JAX PyTree compatibility."""
    children: tuple = ()
    aux_data: dict = dict(self.__dict__.items())
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data: dict, _: tuple) -> BaseSamplerConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(**aux_data)

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
    """Flatten the dataclass for JAX PyTree compatibility."""
    children: tuple = ()
    aux_data: dict = dict(self.__dict__.items())
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data: dict, _: tuple) -> SMCConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(**aux_data)

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
  population: PopulationSequences
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
class SMCOutput(SamplerOutputProtocol):  # Implements the output protocol
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
    """Flatten the dataclass for JAX PyTree compatibility."""
    children: tuple = ()
    aux_data: dict = self.__dict__
    return (children, aux_data)

  @classmethod
  def tree_unflatten(
    cls: type[ExchangeConfig],
    aux_data: dict,
    _children: tuple,
  ) -> ExchangeConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(**aux_data)


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
    """Flatten the dataclass for JAX PyTree compatibility."""
    children: tuple = ()
    aux_data: dict = self.__dict__
    return (children, aux_data)

  @classmethod
  def tree_unflatten(
    cls: type[PRSMCStepConfig],
    aux_data: dict,
    _children: tuple,
  ) -> PRSMCStepConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(**aux_data)


@dataclass(frozen=True)
class ParallelReplicaConfig(BaseSamplerConfig):
  """Configuration for parallel replica SMC simulation."""

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
      msg = f"Length of island_betas ({len(self.island_betas)}) must match n_islands ({self.n_islands})."
      raise ValueError(msg)
    if not all(0.0 <= beta <= 1.0 for beta in self.island_betas):
      msg = "All island_betas must be in [0.0, 1.0]."
      raise ValueError(msg)
    if not isinstance(self.step_config, PRSMCStepConfig):
      msg = "step_config must be a PRSMCStepConfig instance."
      raise TypeError(msg)

  def tree_flatten(self) -> tuple[tuple, dict]:
    """Flatten the dataclass for JAX PyTree compatibility."""
    children: tuple = ()
    aux_data: dict = dict(self.__dict__.items())
    return (children, aux_data)

  @classmethod
  def tree_unflatten(cls, aux_data: dict, _: tuple) -> ParallelReplicaConfig:
    """Unflatten the dataclass for JAX PyTree compatibility."""
    return cls(**aux_data)

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
  population: PopulationSequences
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
    return cls(*children)

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


jax.tree_util.register_pytree_node_class(ExchangeConfig)
jax.tree_util.register_pytree_node_class(PRSMCStepConfig)
jax.tree_util.register_pytree_node_class(ParallelReplicaConfig)
jax.tree_util.register_pytree_node_class(IslandState)
jax.tree_util.register_pytree_node_class(PRSMCCarryState)
jax.tree_util.register_pytree_node_class(ParallelReplicaSMCOutput)
jax.tree_util.register_pytree_node_class(MemoryConfig)
jax.tree_util.register_pytree_node_class(SMCConfig)
jax.tree_util.register_pytree_node_class(SMCCarryState)
jax.tree_util.register_pytree_node_class(SMCOutput)
