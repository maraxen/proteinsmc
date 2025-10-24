"""Base configuration and protocol definitions for samplers in the proteinsmc package."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from blackjax.smc.base import SMCState as BaseSMCState
from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride as InnerMCMCState
from blackjax.smc.partial_posteriors_path import PartialPosteriorsSMCState
from blackjax.smc.pretuning import StateWithParameterOverride as PretuningSMCState
from blackjax.smc.tempered import TemperedSMCState
from flax import struct
from jax.sharding import Mesh

from proteinsmc.models.memory import MemoryConfig

if TYPE_CHECKING:
  from blackjax.base import State as BlackjaxState
  from blackjax.mcmc.random_walk import RWState
  from jaxtyping import PRNGKeyArray

  from proteinsmc.models.types import EvoSequence
  from proteinsmc.utils.annealing import AnnealingConfig

from proteinsmc.models.fitness import FitnessEvaluator

BlackjaxSMCState = (
  BaseSMCState | TemperedSMCState | InnerMCMCState | PartialPosteriorsSMCState | PretuningSMCState
)


@dataclass(frozen=True)
class BaseSamplerConfig:
  """Base configuration for samplers.

  All sampler configurations should inherit from this.
  """

  prng_seed: int = field(default=42)
  sampler_type: str = field(default="unknown")

  device_mesh_shape: tuple[int, ...] | None = field(default=None)
  """Shape of the device mesh (e.g., (8,) for 1D, (4, 2) for 2D).
  If None, autodetects all available devices and creates a 1D mesh.
  """
  axis_names: tuple[str, ...] | None = field(default=None)
  """Names for the mesh axes (e.g., ('devices',)). Must match the length
  of device_mesh_shape if provided.
  """

  mesh: Mesh = field(init=False, repr=False, compare=False, metadata={"pytree_node": False})
  """The live JAX device mesh. Created in __post_init__."""

  memory_config: MemoryConfig = field(default_factory=MemoryConfig)

  fitness_evaluator: FitnessEvaluator = field(
    kw_only=True,
  )
  """Fitness evaluator to assess the quality of sampled sequences."""

  seed_sequence: str | Sequence[str] = field(default="")
  num_samples: int | Sequence[int] = field(default=100)
  n_states: int = field(default=20)
  mutation_rate: float | Sequence[float] = field(default=0.1)
  diversification_ratio: float | Sequence[float] = field(default=0.0)
  sequence_type: Literal["protein", "nucleotide"] | Sequence[Literal["protein", "nucleotide"]] = (
    field(
      default="protein",
    )
  )

  combinations_mode: Literal["zip", "product"] | Sequence[Literal["zip", "product"]] = field(
    default="zip",
  )
  annealing_config: AnnealingConfig | None = field(default=None, kw_only=True)

  def _validate_types(self) -> None:
    """Check types of the fields."""
    if self.fitness_evaluator is not None and not isinstance(
      self.fitness_evaluator,
      FitnessEvaluator,
    ):
      msg = "fitness_evaluator must be a FitnessEvaluator instance or None."
      raise TypeError(msg)
    if not isinstance(self.seed_sequence, (str, Sequence)):
      msg = "seed_sequence must be a string or a sequence of strings."
      raise TypeError(msg)
    # Check num_samples - must be int or Sequence, but not string
    if not isinstance(self.num_samples, int | Sequence) or isinstance(self.num_samples, str):
      msg = "num_samples must be an integer or a sequence of integers."
      raise TypeError(msg)
    if not isinstance(self.n_states, (int, Sequence)):
      msg = "n_states must be an integer or a sequence of integers."
      raise TypeError(msg)
    if not isinstance(self.mutation_rate, (float, Sequence)):
      msg = "mutation_rate must be a float or a sequence of floats."
      raise TypeError(msg)
    if not isinstance(self.diversification_ratio, (float, Sequence)):
      msg = "diversification_ratio must be a float or a sequence of floats."
      raise TypeError(msg)
    if not isinstance(
      self.sequence_type,
      (str, Sequence),
    ) or (
      isinstance(self.sequence_type, str) and self.sequence_type not in ("protein", "nucleotide")
    ):
      msg = "sequence_type must be 'protein', 'nucleotide', or a sequence of these."
      raise TypeError(msg)
    if not isinstance(
      self.combinations_mode,
      str | Sequence,
    ) or (
      isinstance(self.combinations_mode, str) and self.combinations_mode not in ("zip", "product")
    ):
      msg = "combinations_mode must be 'zip', 'product', or a sequence of these."
      raise TypeError(msg)
    if not isinstance(self.memory_config, MemoryConfig):
      msg = "memory_config must be a MemoryConfig instance."
      raise TypeError(msg)

  def _check_values(self) -> None:
    """Check values of the fields."""
    if isinstance(self.num_samples, int) and self.num_samples <= 0:
      msg = "num_samples must be positive."
      raise ValueError(msg)
    if isinstance(self.n_states, int) and self.n_states <= 0:
      msg = "n_states must be positive."
      raise ValueError(msg)
    if isinstance(self.mutation_rate, float) and not (0.0 <= self.mutation_rate <= 1.0):
      msg = "mutation_rate must be in [0.0, 1.0]."
      raise ValueError(msg)
    if isinstance(self.diversification_ratio, float) and not (
      0.0 <= self.diversification_ratio <= 1.0
    ):
      msg = "diversification_ratio must be in [0.0, 1.0]."
      raise ValueError(msg)

  def __post_init__(self) -> None:
    """Validate the experiment configuration."""
    self._validate_types()
    self._check_values()
    object.__setattr__(self, "mesh", self._initialize_device_mesh())

  def _initialize_device_mesh(self) -> Mesh:
    """Create the JAX device mesh based on the configuration."""
    devices = jax.devices()
    n_devices = len(devices)

    # 1. Autodetect case
    if self.device_mesh_shape is None:
      names = self.axis_names or ("devices",)
      if len(names) != 1:
        msg = "Default mesh is 1D, so axis_names must have one element if provided."
        raise ValueError(msg)
      return Mesh(devices, axis_names=names)

    # 2. User-provided shape case
    if math.prod(self.device_mesh_shape) != n_devices:
      msg = (
        f"Mesh shape {self.device_mesh_shape} requires {math.prod(self.device_mesh_shape)} "
        f"devices, but {n_devices} are available."
      )
      raise ValueError(msg)

    names = self.axis_names
    if names is None:
      msg = "You must provide axis_names if you specify a device_mesh_shape."
      raise ValueError(msg)
    if len(names) != len(self.device_mesh_shape):
      msg = (
        f"Length of axis_names {names} must match length of device_mesh_shape "
        f"{self.device_mesh_shape}."
      )
      raise ValueError(msg)

    return Mesh(np.array(devices).reshape(self.device_mesh_shape), axis_names=names)

  @property
  def additional_config_fields(self) -> dict[str, str]:
    """Return additional fields for the configuration that are not part of the PyTree."""
    return {}


@struct.dataclass
class SamplerOutput:
  """Unified output structure for all samplers.

  All fields default to empty arrays to ensure msgpack compatibility.
  Samplers populate only the fields relevant to their algorithm.

  Core fields (always populated):
      step: Current step number
      sequences: Sampled sequences
      fitness: Fitness values
      key: RNG key state

  Common metrics (populated when available):
      weights: Particle weights (SMC)
      log_likelihood_increment: Log likelihood increment (SMC)

  SMC-specific:
      ancestors: Ancestor indices for resampling
      ess: Effective sample size
      update_info: Additional information from BlackJax update step (SMCInfo.update_info)

  Tempered SMC (BlackJax):
      lmbda: Tempering parameter (lambda) for tempered SMC
          Note: Not yet implemented in SMC loop, placeholder for future use

  Annealing:
      beta: Inverse temperature parameter

  PRSMC-specific:
      num_attempted_swaps: Number of replica exchange attempts
      num_accepted_swaps: Number of successful swaps
      migration_island_from: Source island indices
      migration_island_to: Destination island indices
      migration_particle_idx_from: Source particle indices
      migration_particle_idx_to: Destination particle indices
      migration_accepted: Whether each swap was accepted
      migration_log_acceptance_ratio: Log acceptance ratio for each swap

  HMC/NUTS-specific:
      acceptance_probability: Acceptance probability
      num_integration_steps: Number of leapfrog steps

  Computed metrics:
      mean_fitness: Mean fitness across population
      max_fitness: Maximum fitness in population
      log_z_estimate: Log partition function estimate

  """

  # Core fields (always populated)
  step: jax.Array
  sequences: EvoSequence
  fitness: jax.Array
  key: jax.Array

  # Common metrics
  weights: jax.Array = struct.field(default_factory=lambda: jnp.array([]))
  log_likelihood_increment: jax.Array = struct.field(default_factory=lambda: jnp.array(0.0))

  # SMC-specific
  ancestors: jax.Array = struct.field(default_factory=lambda: jnp.array([], dtype=jnp.int32))
  ess: jax.Array = struct.field(default_factory=lambda: jnp.array(0.0))
  update_info: jax.Array = struct.field(
    default_factory=lambda: jnp.array([]),
    metadata={
      "help": (
        "BlackJax SMCInfo.update_info - additional info from update step. "
        "NOTE: BlackJax returns this as a NamedTuple which is NOT msgpack-serializable. "
        "Samplers should convert relevant fields to arrays before storing. "
        "Default empty array indicates no update_info was captured."
      )
    },
  )

  # Tempered SMC (BlackJax) - placeholder for future implementation
  lmbda: jax.Array = struct.field(
    default_factory=lambda: jnp.array(-1.0),
    metadata={"help": "Tempering parameter for tempered SMC (not yet implemented)"},
  )

  # Annealing
  beta: jax.Array = struct.field(default_factory=lambda: jnp.array(-1.0))

  # PRSMC-specific
  num_attempted_swaps: jax.Array = struct.field(
    default_factory=lambda: jnp.array(0, dtype=jnp.int32)
  )
  num_accepted_swaps: jax.Array = struct.field(
    default_factory=lambda: jnp.array(0, dtype=jnp.int32)
  )
  migration_island_from: jax.Array = struct.field(
    default_factory=lambda: jnp.array([], dtype=jnp.int32)
  )
  migration_island_to: jax.Array = struct.field(
    default_factory=lambda: jnp.array([], dtype=jnp.int32)
  )
  migration_particle_idx_from: jax.Array = struct.field(
    default_factory=lambda: jnp.array([], dtype=jnp.int32)
  )
  migration_particle_idx_to: jax.Array = struct.field(
    default_factory=lambda: jnp.array([], dtype=jnp.int32)
  )
  migration_accepted: jax.Array = struct.field(
    default_factory=lambda: jnp.array([], dtype=jnp.bool_)
  )
  migration_log_acceptance_ratio: jax.Array = struct.field(
    default_factory=lambda: jnp.array([], dtype=jnp.float32)
  )

  # HMC/NUTS-specific
  acceptance_probability: jax.Array = struct.field(default_factory=lambda: jnp.array(0.0))
  num_integration_steps: jax.Array = struct.field(
    default_factory=lambda: jnp.array(0, dtype=jnp.int32)
  )

  # Computed metrics
  mean_fitness: jax.Array = struct.field(default_factory=lambda: jnp.array(0.0))
  max_fitness: jax.Array = struct.field(default_factory=lambda: jnp.array(0.0))
  log_z_estimate: jax.Array = struct.field(default_factory=lambda: jnp.array(0.0))


class SamplerOutputProtocol(Protocol):
  """Protocol for sampler output dataclasses."""


@struct.dataclass
class SamplerState:
  """Immutable state for samplers, compatible with JAX transformations.

  This is a PyTreeNode that can be passed through jax.jit, jax.lax.scan, etc.
  Use the `.replace()` method to create modified copies.
  """

  sequence: EvoSequence
  key: PRNGKeyArray
  blackjax_state: BlackjaxState | BlackjaxSMCState | RWState | None = None
  step: jax.Array = struct.field(default_factory=lambda: jax.numpy.array(0, dtype=jax.numpy.int32))
  update_parameters: dict[str, jax.Array] = struct.field(default_factory=dict)
  additional_fields: dict[str, jax.Array] = struct.field(default_factory=dict)


def config_to_jax(config: BaseSamplerConfig) -> dict[str, jax.Array]:
  """Convert configuration fields to JAX arrays for JIT compatibility.

  Note: String fields are excluded as JAX does not support string arrays.
  """
  jax_config = {}
  for field_name, field_value in config.__dict__.items():
    if isinstance(field_value, str):
      # Skip string fields - JAX doesn't support string arrays
      continue
    if isinstance(field_value, (int, float, bool)):
      jax_config[field_name] = jax.numpy.array(field_value)
    else:
      jax_config[field_name] = field_value
  return jax_config
