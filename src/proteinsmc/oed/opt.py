"""OED design optimization module."""

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Float, PRNGKeyArray

from proteinsmc.oed.structs import OEDDesign, OEDPredictedVariables


def perturb_design(design: OEDDesign, param_name: str, epsilon: float) -> OEDDesign:
  """Perturb a parameter in the design."""
  val = getattr(design, param_name)
  new_val = val + epsilon
  return design.replace(**{param_name: new_val})


def calculate_fim_determinant(
  design: OEDDesign,
  surrogate_model: Callable[[OEDDesign], OEDPredictedVariables],
  key: PRNGKeyArray,
) -> Float:
  """Calculate the determinant of the Fisher Information Matrix for a design.

  Args:
      design: The experimental design to evaluate.
      surrogate_model: A surrogate model that predicts experimental outcomes.
      key: JAX PRNG key.

  Returns:
      Determinant of the Fisher Information Matrix (higher is better).

  """
  del key
  # Compute sensitivities using finite differences
  epsilon = 1e-4
  base_prediction = surrogate_model(design)

  # Initialize FIM
  n_params = 6  # N, K, q, population_size, mutation_rate, diversification_ratio
  fim = jnp.zeros((n_params, n_params))

  # Compute sensitivities for each parameter
  # This is a simplified approach; a full implementation would use automatic differentiation
  param_names = ["N", "K", "q", "population_size", "mutation_rate", "diversification_ratio"]

  for i, param_i in enumerate(param_names):
    for j, param_j in enumerate(param_names):
      if i <= j:  # FIM is symmetric
        # Perturb parameters
        design_i_plus = perturb_design(design, param_i, epsilon)
        design_j_plus = perturb_design(design, param_j, epsilon)

        # Get predictions
        pred_i_plus = surrogate_model(design_i_plus)
        pred_j_plus = surrogate_model(design_j_plus)

        # Compute partial derivatives
        d_info_i = (pred_i_plus.information_gain - base_prediction.information_gain) / epsilon
        d_info_j = (pred_j_plus.information_gain - base_prediction.information_gain) / epsilon

        # Update FIM
        fim = fim.at[i, j].set(d_info_i * d_info_j)
        if i != j:
          fim = fim.at[j, i].set(fim[i, j])  # Symmetry

  # Return determinant
  return jnp.linalg.det(fim)
