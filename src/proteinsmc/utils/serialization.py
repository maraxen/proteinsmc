"""Serialization utilities for the proteinsmc package."""

from __future__ import annotations

from typing import cast

import jax.numpy as jnp
from proteinsmc.models.sampler_base import BaseSamplerConfig, SamplerOutput

def create_sampler_output_skeleton(config: BaseSamplerConfig) -> SamplerOutput:
  """Create a skeleton SamplerOutput for deserialization.

  Args:
    config: The experiment configuration.

  Returns:
    A SamplerOutput instance with correct shapes and dtypes.
  """
  # 1. Determine population size
  # Check for explicit population_size (SMC)
  if hasattr(config, "population_size"):
      num_samples = getattr(config, "population_size")
  else:
      num_samples = config.num_samples

  if not isinstance(num_samples, int):
    try:
      num_samples = int(num_samples)
    except (TypeError, ValueError):
      # Handle JAX array or sequence
      if hasattr(num_samples, "item"):
        num_samples = int(num_samples.item())  # type: ignore
      elif hasattr(num_samples, "__getitem__"):
        num_samples = int(num_samples[0])  # type: ignore
      else:
        num_samples = 100  # Fallback

  # 2. Determine sequence dimensions
  alphabet_size = config.n_states
  if not isinstance(alphabet_size, int):
    if hasattr(alphabet_size, "item"):
      alphabet_size = int(alphabet_size.item())  # type: ignore
    else:
      alphabet_size = 20  # Fallback

  # Sequence length from seed_sequence
  seed = config.seed_sequence
  seq_len = 0
  if isinstance(seed, str):
    seq_len = len(seed)
  elif hasattr(seed, "shape"):
    shape = getattr(seed, "shape")
    if len(shape) == 2:  # (L, A)
      seq_len = shape[0]
    elif len(shape) == 3:  # (Batch, L, A)
      seq_len = shape[1]
  elif isinstance(seed, (list, tuple)):
    if len(seed) > 0 and isinstance(seed[0], str):
      seq_len = len(seed[0])

  if seq_len == 0:
    seq_len = 1  # Fallback

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
