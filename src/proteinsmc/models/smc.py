"""Type aliases for nucleotide and protein sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (  # Added Any, Callable for clarity
  TYPE_CHECKING,
  Any,
)

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

if TYPE_CHECKING:
  from jaxtyping import PRNGKeyArray

from .annealing import AnnealingScheduleConfig
from .base import BaseSamplerConfig, SamplerOutputProtocol

PopulationNucleotideSequences = Int[Array, "population_size nucleotide_sequence_length"]
PopulationProteinSequences = Int[Array, "population_size protein_sequence_length"]
PopulationSequences = PopulationNucleotideSequences | PopulationProteinSequences
PopulationMetrics = Float[Array, "population_size"]
PopulationBools = Bool[Array, "population_size"]
StackedPopulationMetrics = Float[PopulationMetrics, "population_size combine_funcs"]
"""Data structures for SMC sampling algorithms."""


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


jax.tree_util.register_pytree_node_class(SMCConfig)
jax.tree_util.register_pytree_node_class(SMCCarryState)
jax.tree_util.register_pytree_node_class(SMCOutput)
