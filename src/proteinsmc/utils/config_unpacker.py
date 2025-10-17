"""Utilities for unpacking configuration objects into function arguments.

This module provides wrapper functions that automatically extract and pass
keyword arguments from configuration objects to underlying functions.
"""

import functools
import inspect
from typing import Any, Callable, TypeVar

from proteinsmc.models import BaseSamplerConfig

F = TypeVar("F", bound=Callable[..., Any])


def with_config(func: F) -> F:
  """Wrap a function to pass keyword arguments from BaseSamplerConfig.

  Inspects the wrapped function's signature and automatically passes matching
  attributes from the BaseSamplerConfig object as keyword arguments.

  Args:
    func: The function to wrap. Must accept a `sampler_config` parameter
      of type BaseSamplerConfig.

  Returns:
    The wrapped function with automatic config unpacking.

  Example:
    >>> @with_sampler_config
    ... def my_sampler(sampler_config: BaseSamplerConfig, n_particles: int):
    ...   print(f"Using {n_particles} particles")
    >>> config = SMCConfig(n_particles=1000, ...)
    >>> my_sampler(config)  # Automatically passes n_particles=1000

  """
  sig = inspect.signature(func)
  param_names = set(sig.parameters.keys()) - {"sampler_config"}

  @functools.wraps(func)
  def wrapper(sampler_config: BaseSamplerConfig, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Unpack config attributes into function arguments."""
    # Extract matching attributes from config
    config_kwargs = {
      name: getattr(sampler_config, name)
      for name in param_names
      if hasattr(sampler_config, name) and name not in kwargs
    }
    # Merge with explicit kwargs (explicit takes precedence)
    all_kwargs = {**config_kwargs, **kwargs}
    return func(sampler_config, *args, **all_kwargs)

  return wrapper  # type: ignore[return-value]
