"""OED design optimization module."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Float, PRNGKeyArray

from proteinsmc.oed.structs import OEDDesign, OEDPredictedVariables


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


def recommend_next_design(
  key: PRNGKeyArray,
  design_history: list[tuple[OEDDesign, OEDPredictedVariables]],
  parameter_bounds: dict[str, tuple[float, float]],
  n_candidates: int = 100,
  surrogate_type: str = "gaussian_process",
) -> OEDDesign:
  """Recommend the next experimental design to maximize information gain.

  Args:
      key: JAX PRNG key.
      design_history: History of previous designs and their outcomes.
      parameter_bounds: Bounds for each design parameter.
      n_candidates: Number of candidate designs to evaluate.
      surrogate_type: Type of surrogate model to use.

  Returns:
      Recommended design for the next experiment.

  """
  # Train surrogate model on design history
  surrogate_model = train_surrogate_model(design_history, surrogate_type)

  # Generate candidate designs
  key, subkey = jax.random.split(key)
  candidate_designs = generate_candidate_designs(subkey, parameter_bounds, n_candidates)

  # Evaluate designs using FIM determinant
  fim_values = jnp.array(
    [calculate_fim_determinant(design, surrogate_model, key) for design in candidate_designs]
  )

  # Select best design
  best_idx = jnp.argmax(fim_values)
  return candidate_designs[best_idx]


def train_surrogate_model(
  design_history: list[tuple[OEDDesign, OEDPredictedVariables]],
  surrogate_type: str = "gaussian_process",
) -> Callable[[OEDDesign], OEDPredictedVariables]:
  """Train a surrogate model on design history.

  Args:
      design_history: History of previous designs and their outcomes.
      surrogate_type: Type of surrogate model to use.

  Returns:
      Trained surrogate model that predicts experimental outcomes.

  """
  # Extract design parameters and outcomes
  design_variables = jnp.array(
    [
      [d.N, d.K, d.q, d.population_size, d.mutation_rate, d.diversification_ratio]
      for d, _ in design_history
    ]
  )

  outcomes = jnp.array(
    [
      [
        v.information_gain,
        v.barrier_crossing_frequency,
        v.final_sequence_entropy,
        v.jsd_from_original_population,
      ]
      for _, v in design_history
    ]
  )

  # TODO: Implement actual surrogate model training (GP, random forest, etc.)
  # For now, return a simple placeholder function
  def surrogate_prediction(design: OEDDesign) -> OEDPredictedVariables:
    # Placeholder implementation - would be replaced with actual prediction
    return OEDPredictedVariables(
      information_gain=jnp.array(0.5),
      barrier_crossing_frequency=jnp.array(0.1),
      final_sequence_entropy=jnp.array(0.8),
      jsd_from_original_population=jnp.array(0.3),
    )

  return surrogate_prediction


def perturb_design(design: OEDDesign, param_name: str, epsilon: float) -> OEDDesign:
  """Create a perturbed version of a design for sensitivity analysis.

  Args:
      design: Original design.
      param_name: Parameter to perturb.
      epsilon: Perturbation amount.

  Returns:
      Perturbed design.

  """
  design_dict = design.__dict__.copy()

  # Handle integer parameters specially
  if param_name in ["N", "K", "q", "population_size"]:
    design_dict[param_name] = int(design_dict[param_name] * (1 + epsilon))
  else:
    design_dict[param_name] = design_dict[param_name] * (1 + epsilon)

  return OEDDesign(**design_dict)
