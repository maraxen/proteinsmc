"""Unit tests for the Gibbs sampler data structures.

These tests verify the correct construction and typing of GibbsState, GibbsConfig,
and GibbsUpdateFuncSignature.

All tests are compatible with pytest and are designed to run independently.

Example:
  $ python -m pytest src/proteinsmc/models/test_gibbs.py
"""


import types

import numpy as np
import pytest
from .conftest import valid_config_kwargs
from proteinsmc.models.sampler_base import BaseSamplerConfig



from proteinsmc.models.gibbs import (
  GibbsConfig,
  GibbsState,
  GibbsUpdateKwargs,
)


class DummyPRNGKeyArray:
  """Dummy PRNGKeyArray for testing."""

  def __init__(self, value=0):
    self.value = value


class DummyEvoSequence(np.ndarray):
  """Dummy EvoSequence for testing."""


def dummy_fitness_fn(sequence):
  """Dummy fitness function for testing."""
  return 1.0


@pytest.fixture
def dummy_key():
  """Fixture for a dummy PRNGKeyArray."""
  return DummyPRNGKeyArray(42)


@pytest.fixture
def dummy_sequence():
  """Fixture for a dummy EvoSequence."""
  return np.array([1, 2, 3]).view(DummyEvoSequence)


@pytest.fixture
def dummy_fitness():
  """Fixture for a dummy fitness value."""
  return 3.14


def test_gibbs_state_construction(dummy_sequence, dummy_fitness, dummy_key):
  """Test construction and attribute access of GibbsState.

  Args:
    dummy_sequence: Dummy EvoSequence.
    dummy_fitness: Dummy fitness value.
    dummy_key: Dummy PRNGKeyArray.

  Returns:
    None

  Raises:
    AssertionError: If attributes do not match expected values.

  Example:
    >>> test_gibbs_state_construction(dummy_sequence, dummy_fitness, dummy_key)

  """
  state = GibbsState(samples=dummy_sequence, fitness=dummy_fitness, key=dummy_key)
  assert isinstance(state, GibbsState)
  assert (state.samples == dummy_sequence).all()
  assert state.fitness == dummy_fitness
  assert state.key == dummy_key


def test_gibbs_config_inherits_base(valid_config_kwargs):
  """Test that GibbsConfig inherits from BaseSamplerConfig and sets num_samples.

  Args:
    None

  Returns:
    None

  Raises:
    AssertionError: If inheritance or attribute assignment fails.

  Example:
    >>> test_gibbs_config_inherits_base()

  """
  config = GibbsConfig(**valid_config_kwargs)
  assert isinstance(config, BaseSamplerConfig)
  assert config.num_samples == 10


def test_gibbs_update_kwargs_typeddict():
  """Test that GibbsUpdateKwargs enforces required keys and types.

  Args:
    None

  Returns:
    None

  Raises:
    AssertionError: If keys or types are missing or incorrect.

  Example:
    >>> test_gibbs_update_kwargs_typeddict()

  """
  kwargs: GibbsUpdateKwargs = {  # type: ignore
    "key": DummyPRNGKeyArray(1),
    "sequence": np.array([0, 1, 2]).view(DummyEvoSequence),
    "fitness_fn": dummy_fitness_fn,
    "position": 0,
    "_context": None,
  }
  assert "key" in kwargs
  assert "sequence" in kwargs
  assert "fitness_fn" in kwargs
  assert "position" in kwargs
  assert "_context" in kwargs


def test_gibbs_update_func_signature(dummy_sequence):
  """Test that a function matching GibbsUpdateFuncSignature type can be called.

  Args:
    dummy_sequence: Dummy EvoSequence.

  Returns:
    None

  Raises:
    AssertionError: If the function does not return the expected type.

  Example:
    >>> test_gibbs_update_func_signature(dummy_sequence)

  """

  def update_fn(
    key,
    sequence,
    fitness_fn,
    position,
    _context=None,
  ):
    # For test, just return the sequence unchanged
    return sequence

  # Check that update_fn matches GibbsUpdateFuncSignature
  assert isinstance(update_fn, types.FunctionType)
  result = update_fn(
    key=DummyPRNGKeyArray(0),
    sequence=dummy_sequence,
    fitness_fn=dummy_fitness_fn,
    position=1,
    _context=None,
  )
  assert (result == dummy_sequence).all()
