"""Tests for the runner module with JAX array inputs."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from unittest.mock import MagicMock, Mock, patch

import jax
import jax.numpy as jnp

from proteinsmc.models import SMCConfig
from proteinsmc.runner import SAMPLER_REGISTRY, run_experiment


class TestRunExperimentWithJAXInputs:
  """Test the run_experiment function with JAX array inputs."""

  @patch("proteinsmc.runner.initialize_sampler_state")
  @patch("proteinsmc.runner.get_fitness_function")
  @patch("proteinsmc.runner.get_annealing_function")
  @patch("proteinsmc.runner._setup_writer_callback")
  def test_run_experiment_with_jax_arrays(
    self,
    mock_setup_writer: Mock,
    mock_get_annealing_function: Mock,
    mock_get_fitness_function: Mock,
    mock_initialize_sampler_state: Mock,
    basic_smc_config: SMCConfig,
  ) -> None:
    """Test that run_experiment can handle jax.Array inputs in the config."""
    # Setup mocks
    mock_fitness_fn = Mock()
    mock_annealing_fn = Mock()
    mock_get_fitness_function.return_value = (jax.random.PRNGKey(0), mock_fitness_fn)
    mock_get_annealing_function.return_value = mock_annealing_fn

    # Mock the writer setup
    mock_writer = Mock()
    mock_io_callback = Mock()
    mock_setup_writer.return_value = (mock_writer, mock_io_callback)

    # Mock the sampler functions
    mock_initial_state = Mock()
    mock_initialize_sampler_state.return_value = mock_initial_state
    mock_final_state = Mock()
    mock_metric = MagicMock()
    mock_metric.shape = (10,)
    mock_item = Mock()
    mock_item.ndim = 0
    mock_item.__float__ = lambda self: 0.0
    mock_metric.__getitem__.return_value = mock_item
    mock_outputs = {"metric1": mock_metric, "metric2": mock_metric}

    mock_run_fn = Mock(return_value=(mock_final_state, mock_outputs))

    # Create a config with jax.Array inputs
    seed_sequence_array = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    mutation_rate_array = jnp.array(0.1)

    jax_config = replace(
      basic_smc_config,
      seed_sequence=seed_sequence_array,
      mutation_rate=mutation_rate_array,
    )

    with patch.dict(SAMPLER_REGISTRY, {"smc": {"config_cls": SMCConfig, "run_fn": mock_run_fn}}):
      with tempfile.TemporaryDirectory() as tmpdir:
        run_experiment(jax_config, tmpdir, seed=42)

    # Verify that initialize_sampler_state was called with the correct JAX arrays
    mock_initialize_sampler_state.assert_called_once()
    _, kwargs = mock_initialize_sampler_state.call_args
    assert jnp.array_equal(kwargs["seed_sequence"], seed_sequence_array)
    assert jnp.array_equal(kwargs["mutation_rate"], mutation_rate_array)
