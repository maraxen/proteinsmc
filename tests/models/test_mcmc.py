"""Unit tests for MCMC sampler data model.

Tests cover initialization and edge cases for MCMC sampler config/model.
"""
import pytest
from proteinsmc.models import mcmc


def test_mcmc_config_initialization():
  """Test MCMCConfig initialization with valid arguments.
  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If the config fields do not match expected values.
  Example:
    >>> test_mcmc_config_initialization()
  """
  config = mcmc.MCMCConfig(
    n_steps=15,
    proposal_std=0.05
  )
  assert config.n_steps == 15
  assert config.proposal_std == 0.05
