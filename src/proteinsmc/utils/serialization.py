"""Serialization utilities for the proteinsmc package."""

from __future__ import annotations

from typing import cast

import jax.numpy as jnp

from proteinsmc.models.sampler_base import BaseSamplerConfig, SamplerOutput


def _determine_population_size(config: BaseSamplerConfig) -> int:
  """Determine population size from config."""
  num_samples = config.population_size if hasattr(config, "population_size") else config.num_samples

  if isinstance(num_samples, int):
    return num_samples

  try:
    return int(cast("int", num_samples))
  except (TypeError, ValueError):
    # Handle JAX array or sequence
    if hasattr(num_samples, "item"):
      return int(num_samples.item())  # type: ignore  # noqa: PGH003
    if hasattr(num_samples, "__getitem__"):
      return int(num_samples[0])  # type: ignore  # noqa: PGH003

  return 100  # Fallback


def _determine_seq_len(config: BaseSamplerConfig) -> int:
  """Determine sequence length from config."""
  seed = config.seed_sequence
  if isinstance(seed, str):
    return len(seed)
  if hasattr(seed, "shape"):
    shape = cast("tuple[int, ...]", seed.shape)
    if len(shape) == 2:  # (L, A)  # noqa: PLR2004
      return shape[0]
    if len(shape) == 3:  # (Batch, L, A)  # noqa: PLR2004
      return shape[1]
  if isinstance(seed, (list, tuple)) and len(seed) > 0 and isinstance(seed[0], str):
    return len(seed[0])

  return 1  # Fallback


def create_sampler_output_skeleton(config: BaseSamplerConfig) -> SamplerOutput:
  """Create a skeleton SamplerOutput for deserialization.

  Args:
    config: The experiment configuration.

  Returns:
    A SamplerOutput instance with correct shapes and dtypes.

  """
  num_samples = _determine_population_size(config)

  # 2. Determine sequence dimensions
  # alphabet_size is unused in skeleton creation but was calculated in original code.
  # We skip it as it's not used for skeleton arrays.

  seq_len = _determine_seq_len(config)

  # 3. Construct skeleton
  # Core fields
  sequences_shape = (num_samples, seq_len)

  sequences = jnp.zeros(sequences_shape, dtype=jnp.int8)
  fitness = jnp.zeros((num_samples,), dtype=jnp.float32)
  step = jnp.array(0, dtype=jnp.int32)
  key = jnp.zeros((2,), dtype=jnp.uint32)

  # Optional fields - logic based on sampler_type
  sampler_type = config.sampler_type.lower()

  # Default empty/scalar values matches SamplerOutput defaults
  weights = jnp.array([])
  ancestors = jnp.array([], dtype=jnp.int32)

  if "smc" in sampler_type or "parallel" in sampler_type:
    weights = jnp.zeros((num_samples,), dtype=jnp.float32)
    ancestors = jnp.zeros((num_samples,), dtype=jnp.int32)

  # Prepare kwargs with overrides
  kwargs = {
    "weights": weights,
    "ancestors": ancestors,
  }

  return SamplerOutput(
    step=step,
    sequences=sequences,
    fitness=fitness,
    key=key,
    **kwargs,
  )
