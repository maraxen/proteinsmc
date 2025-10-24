"""Phase boundary detection module."""

import jax.numpy as jnp
from jaxtyping import Array, Float


def detect_phase_boundaries(
  parameter_values: Array, metric_values: Array, sensitivity_threshold: Float
) -> Float:
  """Detect phase boundaries in evolutionary dynamics.

  Args:
      parameter_values: Array of parameter values (e.g., mutation rates).
      metric_values: Corresponding array of metric values.
      sensitivity_threshold: Threshold for detecting rapid changes.

  Returns:
      List of parameter values where phase boundaries occur.

  """
  # Sort arrays by parameter value
  sorted_idx = jnp.argsort(parameter_values)
  sorted_params = parameter_values[sorted_idx]
  sorted_metrics = metric_values[sorted_idx]

  # Calculate gradient of metric along parameter axis
  grad = jnp.array(jnp.gradient(sorted_metrics, sorted_params))

  # Identify points with gradient magnitude above threshold
  boundary_candidates = jnp.where(jnp.abs(grad) > sensitivity_threshold)[0]

  # Return corresponding parameter values
  return sorted_params[boundary_candidates]
