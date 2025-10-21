"""Base configuration and protocol definitions for samplers in the proteinsmc package."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

import jax
import numpy as np
from blackjax.mcmc.random_walk import RWState
from blackjax.smc.base import SMCState as BaseSMCState
from blackjax.smc.inner_kernel_tuning import StateWithParameterOverride as InnerMCMCState
from blackjax.smc.partial_posteriors_path import PartialPosteriorsSMCState
from blackjax.smc.pretuning import StateWithParameterOverride as PretuningSMCState
from blackjax.smc.tempered import TemperedSMCState
from equinox.nn import State
from jax.sharding import Mesh

from proteinsmc.models.memory import MemoryConfig

if TYPE_CHECKING:
  from blackjax.base import State as BlackjaxState
  from jaxtyping import PRNGKeyArray

  from proteinsmc.models.types import EvoSequence
  from proteinsmc.utils.annealing import AnnealingConfig

from proteinsmc.models.fitness import FitnessEvaluator, StackedFitness

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
  track_lineage: bool = field(default=False)

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
    if not isinstance(self.num_samples, (int, Sequence)):
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
      (str, Sequence),
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


class SamplerOutputProtocol(Protocol):
  """Protocol for sampler output dataclasses."""


from proteinsmc.models.types import UUIDArray


class SamplerState(State):
  """Protocol for sampler state dataclasses."""

  sequence: EvoSequence
  fitness: StackedFitness
  key: PRNGKeyArray
  run_uuid: UUIDArray | None = None
  blackjax_state: BlackjaxState | BlackjaxSMCState | RWState | None = None
  step: jax.Array | None = None
  update_parameters: dict[str, jax.Array] = field(default_factory=dict)
  additional_fields: dict[str, jax.Array] = field(default_factory=dict)


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
