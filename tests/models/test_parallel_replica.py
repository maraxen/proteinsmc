"""Unit tests for ParallelReplicaConfig data model.

Tests cover initialization and edge cases for ParallelReplicaConfig.
"""
import pytest
from proteinsmc.models import parallel_replica


def test_parallel_replica_config_initialization():
  """Test ParallelReplicaConfig initialization with valid arguments.
  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If the config fields do not match expected values.
  Example:
    >>> test_parallel_replica_config_initialization()
  """
  config = parallel_replica.ParallelReplicaConfig(
    n_replicas=4,
    exchange_interval=5
  )
  assert config.n_replicas == 4
  assert config.exchange_interval == 5
