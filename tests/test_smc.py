import pytest
import jax
import jax.numpy as jnp
from jax import random
from unittest.mock import MagicMock

from src.sampling.smc import run_smc_jax
from src.mpnn import mpnn_score
from src.utils.constants import NUCLEOTIDES_CHAR, NUCLEOTIDES_INT_MAP, RES_TO_CODON_CHAR

# Mock MPNN model for testing
class MockMPNNModel:
    def __init__(self):
        self._inputs = {
            "X": jnp.array([[0.0, 0.0, 0.0]]),
            "mask": jnp.array([1.0]),
            "residue_idx": jnp.array([1]),
            "chain_idx": jnp.array([1])
        }

    def _score(self, X, mask, residue_idx, chain_idx, key, S):
        # Mock scoring function, returns a dummy score
        return {"logits": jnp.zeros((S.shape[0], S.shape[1], 21))}

@pytest.fixture
def mock_mpnn_model():
    return MockMPNNModel()

@pytest.fixture
def default_smc_params(mock_mpnn_model):
    return {
        "prng_key_smc_steps": random.PRNGKey(0),
        "initial_population_key": random.PRNGKey(1),
        "initial_population_mutation_rate": 0.1,
        "protein_length": 10,
        "initial_sequence_char": "AAAAAAAAAA",
        "sequence_type": "protein",
        "mu_nuc": 0.01,
        "annealing_schedule_func_py": lambda p, n, b: b * (p / n),
        "beta_max_val": 1.0,
        "annealing_len_val": 10,
        "n_particles": 100,
        "n_smc_steps": 10,
        "mpnn_model_instance": mock_mpnn_model,
        "mpnn_model_is_active_static": True,
        "save_traj": False
    }

def test_run_smc_jax_protein_basic(default_smc_params):
    results = run_smc_jax(**default_smc_params)

    assert isinstance(results, dict)
    assert "final_logZhat" in results
    assert "mean_combined_fitness_per_gen" in results
    assert results["mean_combined_fitness_per_gen"].shape == (default_smc_params["n_smc_steps"],)

def test_run_smc_jax_nucleotide_not_implemented(default_smc_params):
    params = default_smc_params.copy()
    params["sequence_type"] = "nucleotide"
    params["initial_sequence_char"] = "ATGCATGCAT" * 3 # Example nucleotide sequence

    with pytest.raises(ValueError, match="Unsupported sequence_type: nucleotide"):
        run_smc_jax(**params)

def test_run_smc_jax_zero_steps(default_smc_params):
    params = default_smc_params.copy()
    params["n_smc_steps"] = 0
    results = run_smc_jax(**params)

    assert isinstance(results, dict)
    assert results["final_logZhat"] == 0.0
    assert results["mean_combined_fitness_per_gen"].shape == (0,)
    assert results["entropy_per_gen"].shape == (0,)

def test_run_smc_jax_initial_population_mutation_rate(default_smc_params):
    params = default_smc_params.copy()
    params["initial_population_mutation_rate"] = 0.0 # No initial mutation
    results = run_smc_jax(**params)
    # Add assertions to check if initial population is as expected (e.g., no mutations)
    # This might require inspecting the initial particles directly, which is harder
    # given the current return structure. For now, just ensure it runs.
    assert isinstance(results, dict)

def test_run_smc_jax_mpnn_inactive(default_smc_params):
    params = default_smc_params.copy()
    params["mpnn_model_is_active_static"] = False
    results = run_smc_jax(**params)
    # MPNN scores should be zero or NaN if inactive
    assert jnp.all(jnp.isnan(results["mean_mpnn_score_per_gen"])) or jnp.all(results["mean_mpnn_score_per_gen"] == 0.0)

def test_run_smc_jax_invalid_initial_protein_sequence(default_smc_params):
    params = default_smc_params.copy()
    params["initial_sequence_char"] = "AAZ" # Invalid amino acid
    params["protein_length"] = 3
    with pytest.raises(ValueError, match="Failed to generate initial JAX nucleotide template from AA 'Z'"):
        run_smc_jax(**params)