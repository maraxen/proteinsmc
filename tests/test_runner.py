"""Tests for the runner module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import replace


import pytest
import jax.numpy as jnp

from proteinsmc.models import SMCConfig
from proteinsmc.runner import SAMPLER_REGISTRY, run_experiment


class TestSamplerRegistry:
  """Test the sampler registry."""

  def test_registry_has_required_keys(self) -> None:
    """Test that all samplers in the registry have required keys."""
    expected_keys = {"config_cls", "initialize_fn", "run_fn"}
    
    for sampler_type, sampler_def in SAMPLER_REGISTRY.items():
      assert isinstance(sampler_type, str)
      assert isinstance(sampler_def, dict)
      assert set(sampler_def.keys()) == expected_keys
      
      # Check that config_cls is a class
      assert callable(sampler_def["config_cls"])
      
      # Check that functions are callable
      assert callable(sampler_def["initialize_fn"])
      assert callable(sampler_def["run_fn"])

  def test_registry_contains_expected_samplers(self) -> None:
    """Test that the registry contains expected sampler types."""
    expected_samplers = {"smc", "parallel_replica", "gibbs", "mcmc", "hmc", "nuts"}
    assert set(SAMPLER_REGISTRY.keys()) == expected_samplers


class TestRunExperiment:
  """Test the run_experiment function."""

  def test_invalid_sampler_type_raises_error(self, basic_smc_config: SMCConfig) -> None:
    """Test that invalid sampler type raises ValueError."""
    # Create a copy of the config with an invalid sampler type
    invalid_config = replace(basic_smc_config, sampler_type="invalid_sampler")
    
    with tempfile.TemporaryDirectory() as tmpdir:
      with pytest.raises(ValueError, match="Unknown sampler_type"):
        run_experiment(invalid_config, tmpdir, seed=42)

  def test_config_type_mismatch_raises_error(self, basic_smc_config: SMCConfig) -> None:
    """Test that config type mismatch raises TypeError."""
    # Create a copy of the config with a mismatched sampler type
    wrong_config = replace(basic_smc_config, sampler_type="gibbs")
    
    with tempfile.TemporaryDirectory() as tmpdir:
      with pytest.raises(TypeError, match="Configuration object of type"):
        run_experiment(wrong_config, tmpdir, seed=42)

  @patch("proteinsmc.runner.get_fitness_function")
  @patch("proteinsmc.runner.get_annealing_function")
  @patch("proteinsmc.runner.RunManager")
  def test_successful_run_without_auto_tuning(
    self,
    mock_run_manager: Mock,
    mock_get_annealing_function: Mock,
    mock_get_fitness_function: Mock,
    basic_smc_config: SMCConfig,
  ) -> None:
    """Test successful experiment run without auto-tuning."""
    # Setup mocks
    mock_fitness_fn = Mock()
    mock_annealing_fn = Mock()
    mock_get_fitness_function.return_value = mock_fitness_fn
    mock_get_annealing_function.return_value = mock_annealing_fn
    
    # Mock the RunManager context manager
    mock_writer = Mock()
    mock_writer.run_id = "test_run_id"
    mock_run_manager.return_value.__enter__.return_value = mock_writer
    
    # Mock the sampler functions
    mock_initial_state = Mock()
    mock_final_state = Mock()
    mock_metric = MagicMock()
    mock_metric.shape = (10,)
    mock_item = Mock()
    mock_item.ndim = 0
    mock_item.__float__ = lambda self: 0.0
    mock_metric.__getitem__.return_value = mock_item
    mock_outputs = {"metric1": mock_metric, "metric2": mock_metric}
    
    with patch.dict(SAMPLER_REGISTRY, {
      "smc": {
        "config_cls": SMCConfig,
        "initialize_fn": Mock(return_value=mock_initial_state),
        "run_fn": Mock(return_value=(mock_final_state, mock_outputs)),
      }
    }):
      with tempfile.TemporaryDirectory() as tmpdir:
        run_experiment(basic_smc_config, tmpdir, seed=42)
    
    # Verify the functions were called
    mock_get_fitness_function.assert_called()
    mock_get_annealing_function.assert_called_once()
    mock_run_manager.assert_called_once()

  @patch("proteinsmc.runner.auto_tune_chunk_size")
  @patch("proteinsmc.runner.get_fitness_function")
  @patch("proteinsmc.runner.get_annealing_function")
  @patch("proteinsmc.runner.RunManager")
  def test_successful_run_with_auto_tuning(
    self,
    mock_run_manager: Mock,
    mock_get_annealing_function: Mock,
    mock_get_fitness_function: Mock,
    mock_auto_tune_chunk_size: Mock,
    basic_smc_config: SMCConfig,
  ) -> None:
    """Test successful experiment run with auto-tuning enabled."""
    # Enable auto-tuning in config
    auto_tuning_config = basic_smc_config.memory_config.auto_tuning_config
    object.__setattr__(auto_tuning_config, "enable_auto_tuning", True)
    
    # Setup mocks
    mock_fitness_fn = Mock()
    mock_annealing_fn = Mock()
    mock_get_fitness_function.return_value = mock_fitness_fn
    mock_get_annealing_function.return_value = mock_annealing_fn
    mock_auto_tune_chunk_size.return_value = 32
    
    # Mock the RunManager context manager
    mock_writer = Mock()
    mock_writer.run_id = "test_run_id"
    mock_run_manager.return_value.__enter__.return_value = mock_writer
    
    # Mock the sampler functions
    mock_initial_state = Mock()
    mock_final_state = Mock()

    mock_metric = MagicMock()
    mock_metric.shape = (10,)
    mock_item = Mock()
    mock_item.ndim = 0
    mock_item.__float__ = lambda self: 0.0
    mock_metric.__getitem__.return_value = mock_item
    mock_outputs = {"metric1": mock_metric, "metric2": mock_metric}
    
    with patch.dict(SAMPLER_REGISTRY, {
      "smc": {
        "config_cls": SMCConfig,
        "initialize_fn": Mock(return_value=mock_initial_state),
        "run_fn": Mock(return_value=(mock_final_state, mock_outputs)),
      }
    }):
      with tempfile.TemporaryDirectory() as tmpdir:
        run_experiment(basic_smc_config, tmpdir, seed=42)
    
    # Verify auto-tuning was called
    mock_auto_tune_chunk_size.assert_called_once()
    # Verify fitness function was called twice (once without chunk_size, once with)
    assert mock_get_fitness_function.call_count == 2

  def test_experiment_creates_output_directory(self, basic_smc_config: SMCConfig) -> None:
    """Test that experiment creates output directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
      output_dir = Path(tmpdir) / "nonexistent_dir"
      assert not output_dir.exists()
      
      mock_metric = MagicMock()
      mock_metric.shape = (10,)
      mock_item = Mock()
      mock_item.ndim = 0
      mock_item.__float__ = lambda self: 0.0
      mock_metric.__getitem__.return_value = mock_item

      with patch.dict(SAMPLER_REGISTRY, {
        "smc": {
          "config_cls": SMCConfig,
          "initialize_fn": Mock(),

          "run_fn": Mock(return_value=(Mock(), {"metric": mock_metric})),
        }
      }), patch("proteinsmc.runner.RunManager"), \
         patch("proteinsmc.runner.get_fitness_function"), \
         patch("proteinsmc.runner.get_annealing_function"):
        
        run_experiment(basic_smc_config, str(output_dir), seed=42)