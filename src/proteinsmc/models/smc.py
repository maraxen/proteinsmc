"""Type aliases for nucleotide and protein sequences."""

from __future__ import annotations

from dataclasses import dataclass, field

from flax import struct
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from .annealing import AnnealingConfig
from .sampler_base import BaseSamplerConfig, SamplerOutputProtocol

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

  population_size: int = field(default=64)
  """Number of sequences in the population."""
  annealing_schedule: AnnealingConfig = field(kw_only=True)
  sampler_type: str = field(default="smc", init=False)

  def _validate_types(self) -> None:
    """Validate the types of the fields."""
    if not isinstance(self.population_size, int):
      msg = "population_size must be an integer."
      raise TypeError(msg)
    if not isinstance(self.annealing_schedule, AnnealingConfig):
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

  @property
  def additional_config_fields(self) -> dict[str, str]:
    """Return additional fields for the configuration that are not part of the PyTree."""
    return {
      "population_size": "population_size",
      "annealing_schedule": "annealing_schedule",
    }


class SMCState(struct.PyTreeNode):
  """State of the SMC sampler at a single step. Designed to be a JAX PyTree."""

  population: PopulationSequences
  log_weights: PopulationMetrics
  log_likelihood: PopulationMetrics
  beta: float
  key: PRNGKeyArray
  step: int = 0


@dataclass
class SMCOutput(SamplerOutputProtocol):
  """Output of the SMC sampler."""

  mean_combined_fitness_per_gen: Float
  max_combined_fitness_per_gen: Float
  entropy_per_gen: Float
  beta_per_gen: Float
  ess_per_gen: Float
  fitness_components_per_gen: Float
  final_logZhat: Float  # noqa: N815
  final_amino_acid_entropy: Float

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
