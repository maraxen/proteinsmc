"""Manage a full "outer loop" for Optimal Experimental Design (OED)."""

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from proteinsmc.oed.experiment import run_oed_experiment
from proteinsmc.oed.gp import (
  design_to_features,
  features_to_predicted_variables,
  fit_gp_model,
  predict_with_gp_models,
)
from proteinsmc.oed.opt import calculate_fim_determinant
from proteinsmc.oed.structs import OEDDesign, OEDFeatureMode, OEDPredictedVariables
from proteinsmc.oed.tracking import (
  OEDRecordParams,
  add_oed_to_metadata,
  save_oed_checkpoint,
  save_oed_record,
)

# --- Constants and Configuration ---

PARAMETER_BOUNDS = {
  "N": (10, 200),  # Sequence length - ensure meaningful sequences for metrics
  "K": (0, 5),
  "q": (2, 20),
  "population_size": (50, 100000),
  "mutation_rate": (1e-9, 1e-1),
  "diversification_ratio": (0.0, 1.0),
}

FIXED_N_GENERATIONS = 200

# --- OED Core Functions ---


def generate_candidate_designs(
  key: PRNGKeyArray,
  parameter_bounds: dict[str, tuple[float, float]],
  n_candidates: int,
) -> list[OEDDesign]:
  """Generate a list of candidate OEDDesign objects."""
  designs = []
  for _ in range(n_candidates):
    key, subkey = jax.random.split(key)
    n = jax.random.uniform(
      subkey,
      minval=parameter_bounds["N"][0],
      maxval=parameter_bounds["N"][1],
      shape=(),
    )
    key, subkey = jax.random.split(key)
    k = jax.random.uniform(
      subkey,
      minval=parameter_bounds["K"][0],
      maxval=parameter_bounds["K"][1],
      shape=(),
    )
    key, subkey = jax.random.split(key)
    q = jax.random.uniform(
      subkey,
      minval=parameter_bounds["q"][0],
      maxval=parameter_bounds["q"][1],
      shape=(),
    )
    key, subkey = jax.random.split(key)
    population_size = jax.random.uniform(
      subkey,
      minval=parameter_bounds["population_size"][0],
      maxval=parameter_bounds["population_size"][1],
      shape=(),
    )
    key, subkey = jax.random.split(key)
    mutation_rate = jax.random.uniform(
      subkey,
      minval=parameter_bounds["mutation_rate"][0],
      maxval=parameter_bounds["mutation_rate"][1],
      shape=(),
    )
    key, subkey = jax.random.split(key)
    diversification_ratio = jax.random.uniform(
      subkey,
      minval=parameter_bounds["diversification_ratio"][0],
      maxval=parameter_bounds["diversification_ratio"][1],
      shape=(),
    )

    design = OEDDesign(
      N=int(n),
      K=int(k),
      q=int(q),
      population_size=int(population_size),
      n_generations=FIXED_N_GENERATIONS,
      mutation_rate=mutation_rate,
      diversification_ratio=diversification_ratio,
    )
    designs.append(design)
  return designs


def train_surrogate_model(
  design_history: list[tuple[OEDDesign, OEDPredictedVariables]],
  feature_mode: OEDFeatureMode = OEDFeatureMode.ALL,
) -> Callable[[OEDDesign], OEDPredictedVariables]:
  """Train a surrogate model on the design history."""
  x_train = jnp.concatenate(
    [design_to_features(design, mode=feature_mode) for design, _ in design_history], axis=0
  )
  y_train = jnp.array(
    [
      [
        result.information_gain,
        result.barrier_crossing_frequency,
        result.final_sequence_entropy,
        result.jsd_from_original_population,
        result.geometric_fitness_mean,
      ]
      for _, result in design_history
    ]
  )

  models_dict = fit_gp_model(x_train, y_train)

  def predict_surrogate(design: OEDDesign) -> OEDPredictedVariables:
    """Wrap function to make predictions with the trained surrogate model."""
    x_new = design_to_features(design, mode=feature_mode)
    means, variances = predict_with_gp_models(models_dict, x_new)
    return features_to_predicted_variables(means, variances)

  return predict_surrogate


def recommend_next_design(
  key: PRNGKeyArray,
  design_history: list[tuple[OEDDesign, OEDPredictedVariables]],
  parameter_bounds: dict[str, tuple[float, float]],
  n_candidates: int = 100,
  feature_mode: OEDFeatureMode = OEDFeatureMode.ALL,
) -> OEDDesign:
  """Recommend the next experimental design to maximize information gain."""
  surrogate_model = train_surrogate_model(design_history, feature_mode=feature_mode)

  key, subkey = jax.random.split(key)
  candidate_designs = generate_candidate_designs(subkey, parameter_bounds, n_candidates)

  fim_values = jnp.array(
    [calculate_fim_determinant(design, surrogate_model, key) for design in candidate_designs]
  )

  best_idx = jnp.argmax(fim_values)
  return candidate_designs[best_idx]


def main() -> None:
  """Run the OED loop."""
  parser = argparse.ArgumentParser(description="Manage OED loop.")
  parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
  parser.add_argument(
    "--output_dir",
    type=str,
    default="oed_results",
    help="Directory for saving results.",
  )
  parser.add_argument(
    "--num_initial_experiments",
    type=int,
    default=20,
    help="Number of random experiments to seed the model.",
  )
  parser.add_argument(
    "--num_oed_iterations",
    type=int,
    default=100,
    help="Number of OED-driven experiments.",
  )
  parser.add_argument(
    "--feature_mode",
    type=str,
    default="ALL",
    choices=["ALL", "EFFECTIVE_ONLY"],
    help="Feature selection mode for GP.",
  )
  args = parser.parse_args()

  feature_mode = OEDFeatureMode[args.feature_mode]

  key = jax.random.PRNGKey(args.seed)
  design_history = []
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  logging.basicConfig(
    filename=output_dir / "oed_loop.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
  )
  logger = logging.getLogger(__name__)

  # Phase 1: Seeding
  logger.info("--- Starting Seeding Phase ---")
  key, subkey = jax.random.split(key)
  initial_designs = generate_candidate_designs(
    subkey, PARAMETER_BOUNDS, args.num_initial_experiments
  )
  for i, design in enumerate(initial_designs):
    logger.info("Running initial experiment %d/%d", i + 1, args.num_initial_experiments)
    key, subkey = jax.random.split(key)
    result, run_uuid = run_oed_experiment(design, str(output_dir), seed=int(subkey[0]))
    design_history.append((design, result))
    logger.info("Design: %r", design)
    logger.info("Result: %r", result)

    # Enhance SMC run metadata with OED design info
    metadata_file = output_dir / "metadata.json"
    add_oed_to_metadata(metadata_file, design, phase="seeding", iteration=i)

    # Save tracking record
    save_oed_record(
      OEDRecordParams(
        output_dir=output_dir,
        design=design,
        result=result,
        run_uuid=run_uuid,
        iteration=i,
        phase="seeding",
      )
    )

  # Save checkpoint after seeding
  save_oed_checkpoint(output_dir, design_history, 0, args.seed)

  # Phase 2: OED Iterations
  logger.info("--- Starting OED Iteration Phase ---")
  for i in range(args.num_oed_iterations):
    logger.info("Running OED iteration %d/%d", i + 1, args.num_oed_iterations)
    key, subkey = jax.random.split(key)
    next_design = recommend_next_design(
      subkey, design_history, PARAMETER_BOUNDS, n_candidates=100, feature_mode=feature_mode
    )
    key, subkey = jax.random.split(key)
    new_result, run_uuid = run_oed_experiment(next_design, str(output_dir), seed=int(subkey[0]))
    design_history.append((next_design, new_result))
    logger.info("Next Design: %r", next_design)
    logger.info("New Result: %r", new_result)

    # Enhance SMC run metadata with OED design info
    metadata_file = output_dir / "metadata.json"
    add_oed_to_metadata(metadata_file, next_design, phase="optimization", iteration=i)

    # Save tracking record
    save_oed_record(
      OEDRecordParams(
        output_dir=output_dir,
        design=next_design,
        result=new_result,
        run_uuid=run_uuid,
        iteration=i,
        phase="optimization",
      )
    )

    # Save checkpoint every 10 iterations
    if (i + 1) % 10 == 0:
      save_oed_checkpoint(output_dir, design_history, i + 1, args.seed)

    # Log the best result so far
    best_result = max(design_history, key=lambda x: x[1].information_gain)
    logger.info("Best information gain so far: %r", best_result[1].information_gain)

  # Final checkpoint and summary
  save_oed_checkpoint(output_dir, design_history, args.num_oed_iterations, args.seed)
  logger.info("--- OED Loop Completed ---")
  logger.info("Total experiments: %d", len(design_history))


# --- Main Script ---

if __name__ == "__main__":
  main()
