"""Unit tests for AnnealingConfig data model.

Tests cover initialization, type checking, and edge cases for AnnealingConfig.
"""
import pytest
from proteinsmc.models import AnnealingConfig


def test_annealing_config_initialization():
  """Test AnnealingConfig initialization with valid arguments.
  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If the config fields do not match expected values.
  Example:
    >>> test_annealing_config_initialization()
  """
  config = AnnealingConfig(
    annealing_fn="linear",
    beta_max=1.0,
    n_steps=10,
    kwargs={}
  )
  assert config.annealing_fn == "linear"
  assert config.beta_max == 1.0
  assert config.n_steps == 10
  assert isinstance(config.kwargs, dict)

def test_annealing_config_invalid_annealing_fn_type():
  """Test AnnealingConfig raises error when annealing_fn is not a string.
  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If TypeError is not raised for invalid annealing_fn type.
  Example:
    >>> test_annealing_config_invalid_annealing_fn_type()
  """
  with pytest.raises(TypeError):
    AnnealingConfig(
      annealing_fn=123,  # type: ignore[arg-type]
      beta_max=1.0,
      n_steps=10,
      kwargs={}
    )

def test_annealing_config_invalid_beta_max_type():
  """Test AnnealingConfig raises error when beta_max is not a float.
  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If TypeError is not raised for invalid beta_max type.
  Example:
    >>> test_annealing_config_invalid_beta_max_type()
  """
  with pytest.raises(TypeError):
    AnnealingConfig(
      annealing_fn="linear",
      beta_max="high",  # type: ignore[arg-type]
      n_steps=10,
      kwargs={}
    )

def test_annealing_config_invalid_n_steps_type():
  """Test AnnealingConfig raises error when n_steps is not an int.
  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If TypeError is not raised for invalid n_steps type.
  Example:
    >>> test_annealing_config_invalid_n_steps_type()
  """
  with pytest.raises(TypeError):
    AnnealingConfig(
      annealing_fn="linear",
      beta_max=1.0,
      n_steps="ten",  # type: ignore[arg-type]
      kwargs={}
    )

def test_annealing_config_invalid_kwargs_type():
  """Test AnnealingConfig raises error when kwargs is not a dict.
  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If TypeError is not raised for invalid kwargs type.
  Example:
    >>> test_annealing_config_invalid_kwargs_type()
  """
  with pytest.raises(TypeError):
    AnnealingConfig(
      annealing_fn="linear",
      beta_max=1.0,
      n_steps=10,
      kwargs=[]  # type: ignore[arg-type]
    )

def test_annealing_config_multiple_invalid_types():
  """Test AnnealingConfig raises error when multiple arguments have invalid types.
  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If TypeError is not raised for multiple invalid types.
  Example:
    >>> test_annealing_config_multiple_invalid_types()
  """
  with pytest.raises(TypeError):
    AnnealingConfig(
      annealing_fn=123,  # type: ignore[arg-type]
      beta_max="high",   # type: ignore[arg-type]
      n_steps="ten",     # type: ignore[arg-type]
      kwargs=[]          # type: ignore[arg-type]
    )
