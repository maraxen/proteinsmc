"""Data structures for Parallel Replica SMC sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from flax import struct

from .sampler_base import BaseSamplerConfig, SamplerOutputProtocol

if TYPE_CHECKING:
  from blackjax.smc.base import SMCInfo
  from blackjax.smc.base import SMCState as BlackjaxSMCState
  from jaxtyping import Array, Float, Int, PRNGKeyArray

  from proteinsmc.models.fitness import FitnessEvaluator

  from .annealing import AnnealingConfig
  from .smc import Lineage, SMCConfig


@dataclass(frozen=True)
class ExchangeConfig:
  """Configuration for the replica exchange process."""

  n_islands: int = field(default=1)
  """Number of islands in the Parallel Replica SMC."""
  population_size_per_island: int = field(default=64)
  n_exchange_attempts: int = field(default=10)
  exchange_frequency: int = field(default=5)
  fitness_evaluator: FitnessEvaluator = field(kw_only=True)
  sequence_type: str = field(default="protein")
  meta_annealing_schedule: AnnealingConfig = field(kw_only=True)
  track_lineage: bool = field(default=False)


@dataclass(frozen=True)
class ParallelReplicaConfig(BaseSamplerConfig):
  """Configuration for the Parallel Replica SMC sampler."""

  smc_config: SMCConfig = field(kw_only=True)
  n_islands: int = field(default=1)
  n_exchange_attempts: int = field(default=10)
  exchange_frequency: int = field(default=5)
  island_betas: list[float] = field(default_factory=list)
  meta_annealing_schedule: AnnealingConfig = field(kw_only=True)
  track_lineage: bool = field(default=False)
  sampler_type: str = field(default="parallel_replica", init=False)

  def __post_init__(self) -> None:
    """Validate the configuration."""
    super().__post_init__()
    if self.n_islands <= 0:
      msg = "n_islands must be positive."
      raise ValueError(msg)
    if self.smc_config.population_size <= 0:
      msg = "population_size must be positive."
      raise ValueError(msg)
    if self.n_exchange_attempts < 0:
      msg = "n_exchange_attempts must be non-negative."
      raise ValueError(msg)
    if self.exchange_frequency <= 0:
      msg = "exchange_frequency must be positive."
      raise ValueError(msg)
    if len(self.island_betas) != self.n_islands:
      msg = "The number of island_betas must match n_islands."
      raise ValueError(msg)

  @property
  def exchange_config(self) -> ExchangeConfig:
    """Return the exchange configuration."""
    return ExchangeConfig(
      n_islands=self.n_islands,
      n_exchange_attempts=self.n_exchange_attempts,
      exchange_frequency=self.exchange_frequency,
      fitness_evaluator=self.fitness_evaluator,
      sequence_type=self.sequence_type,
      meta_annealing_schedule=self.meta_annealing_schedule,
    )


class IslandState(struct.PyTreeNode):
  """State of a single island in the Parallel Replica SMC sampler."""

  key: PRNGKeyArray
  blackjax_state: BlackjaxSMCState
  beta: Float
  logZ_estimate: Float  # noqa: N815
  mean_fitness: Float
  ess: Float
  step: Int = 0
  lineage: Lineage | None = None
  smc_info: SMCInfo | None = None


class PRSMCState(struct.PyTreeNode):
  """State of the Parallel Replica SMC sampler at a single step."""

  current_overall_state: IslandState
  prng_key: PRNGKeyArray
  total_swaps_attempted: Int
  total_swaps_accepted: Int


@dataclass
class PRSMCOutput(SamplerOutputProtocol):
  """Output of the Parallel Replica SMC sampler."""

  ess_per_island: Array
  mean_fitness_per_island: Array
  max_fitness_per_island: Array
  logZ_increment_per_island: Array  # noqa: N815
  lineage_per_island: Array | None
  meta_beta: Array
  num_accepted_swaps: Array
  num_attempted_swaps: Array

  @property
  def per_gen_stats_metrics(self) -> dict[str, str]:
    """Return mappings for PRSMC per-generation metrics."""
    return {
      "ess_per_island": "ess_per_island",
      "mean_fitness_per_island": "mean_fitness_per_island",
      "max_fitness_per_island": "max_fitness_per_island",
      "logZ_increment_per_island": "logZ_increment_per_island",
      "meta_beta": "meta_beta",
      "num_accepted_swaps": "num_accepted_swaps",
      "num_attempted_swaps": "num_attempted_swaps",
    }

  @property
  def output_type_name(self) -> str:
    """Return the string name of the output type."""
    return "ParallelReplicaSMC"
