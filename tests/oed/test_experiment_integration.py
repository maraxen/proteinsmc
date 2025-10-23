"""Integration test for OED experiment with real SMC runner."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from proteinsmc.oed.experiment import run_oed_experiment
from proteinsmc.oed.structs import OEDDesign, OEDPredictedVariables


def test_run_oed_experiment_small_nk_protein(tmp_path: Path) -> None:
  """Test end-to-end OED experiment with small NK model and protein alphabet.
  
  This is a minimal integration test to verify the entire pipeline:
  - Design -> NK landscape generation
  - NK scoring function registration
  - SMC runner initialization and execution
  - Metrics computation
  - Returns OEDPredictedVariables
  """
  # Create a small design for fast testing
  design = OEDDesign(
    N=10,  # 10 loci for fast computation
    K=2,   # K=2 epistasis
    q=20,  # Protein alphabet (20 amino acids)
    population_size=50,  # Small population for testing
    n_generations=5,     # Just a few SMC steps
    mutation_rate=0.01,
    diversification_ratio=0.1,
  )

  # Run the experiment (this should invoke the full SMC pipeline)
  output_dir = str(tmp_path / "oed_output")
  predicted = run_oed_experiment(design, output_dir, seed=12345)

  # Verify the output structure
  assert isinstance(predicted, OEDPredictedVariables)
  
  # Check that metrics were computed (all should be non-negative)
  assert isinstance(predicted.information_gain, (float, jnp.ndarray))
  assert predicted.information_gain >= 0.0, "Information gain should be non-negative"
  
  assert isinstance(predicted.barrier_crossing_frequency, (float, jnp.ndarray))
  assert predicted.barrier_crossing_frequency >= 0.0, "Barrier frequency should be non-negative"
  
  assert isinstance(predicted.final_sequence_entropy, (float, jnp.ndarray))
  assert predicted.final_sequence_entropy >= 0.0, "Entropy should be non-negative"
  
  assert isinstance(predicted.jsd_from_original_population, (float, jnp.ndarray))
  assert predicted.jsd_from_original_population >= 0.0, "JSD should be non-negative"


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
