"""Tests for I/O utilities including ArrayRecord operations and metadata generation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

from proteinsmc.io import (
  create_metadata_file,
  create_writer_callback,
  get_git_commit_hash,
  read_lineage_data,
)

if TYPE_CHECKING:
  from dataclasses import dataclass


class TestGetGitCommitHash:
  """Test the get_git_commit_hash function."""

  def test_get_git_commit_hash_success(self) -> None:
    """Test successful git commit hash retrieval.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If hash format is incorrect.

    Example:
        >>> test_get_git_commit_hash_success()

    """
    with patch("shutil.which", return_value="/usr/bin/git"):
      with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock(
          stdout="abc123def456\n",
          stderr="",
          returncode=0,
        )
        commit_hash = get_git_commit_hash()

        assert commit_hash == "abc123def456"
        assert isinstance(commit_hash, str)

  def test_get_git_commit_hash_no_git(self) -> None:
    """Test when git is not available.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If return value is not None.

    Example:
        >>> test_get_git_commit_hash_no_git()

    """
    with patch("shutil.which", return_value=None):
      commit_hash = get_git_commit_hash()

      assert commit_hash is None

  def test_get_git_commit_hash_not_in_repo(self) -> None:
    """Test when not in a git repository.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If return value is not None.

    Example:
        >>> test_get_git_commit_hash_not_in_repo()

    """
    with patch("shutil.which", return_value="/usr/bin/git"):
      with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(128, ["git"])
        commit_hash = get_git_commit_hash()

        assert commit_hash is None

  def test_get_git_commit_hash_timeout(self) -> None:
    """Test when git command times out.

    Args:
        None

    Returns:
        None

    Raises:
        AssertionError: If return value is not None.

    Example:
        >>> test_get_git_commit_hash_timeout()

    """
    with patch("shutil.which", return_value="/usr/bin/git"):
      with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["git"], timeout=5)
        commit_hash = get_git_commit_hash()

        assert commit_hash is None


class TestCreateMetadataFile:
  """Test the create_metadata_file function."""

  def test_create_metadata_file_with_dataclass(self, tmp_path: Path) -> None:
    """Test metadata file creation with a dataclass config.

    Args:
        tmp_path: Pytest fixture providing temporary directory.

    Returns:
        None

    Raises:
        AssertionError: If metadata file is not created correctly.

    Example:
        >>> test_create_metadata_file_with_dataclass(tmp_path)

    """
    from dataclasses import dataclass

    @dataclass
    class TestConfig:
      sampler_type: str
      num_steps: int
      mutation_rate: float

    config = TestConfig(sampler_type="smc", num_steps=100, mutation_rate=0.1)

    with patch("proteinsmc.io.get_git_commit_hash", return_value="test_hash"):
      create_metadata_file(config, tmp_path)

    metadata_path = tmp_path / "metadata.json"
    assert metadata_path.exists()

    with metadata_path.open() as f:
      metadata = json.load(f)

    assert metadata["sampler_type"] == "smc"
    assert metadata["git_commit_hash"] == "test_hash"
    assert "timestamp" in metadata
    assert "config" in metadata
    assert metadata["config"]["num_steps"] == 100
    assert metadata["config"]["mutation_rate"] == 0.1

  def test_create_metadata_file_with_plain_object(self, tmp_path: Path) -> None:
    """Test metadata file creation with a non-dataclass object.

    Args:
        tmp_path: Pytest fixture providing temporary directory.

    Returns:
        None

    Raises:
        AssertionError: If metadata file is not created correctly.

    Example:
        >>> test_create_metadata_file_with_plain_object(tmp_path)

    """

    class TestConfig:
      def __init__(self) -> None:
        self.sampler_type = "mcmc"
        self.num_steps = 200

      def __str__(self) -> str:
        return f"TestConfig(sampler_type={self.sampler_type}, num_steps={self.num_steps})"

    config = TestConfig()

    with patch("proteinsmc.io.get_git_commit_hash", return_value=None):
      create_metadata_file(config, tmp_path)

    metadata_path = tmp_path / "metadata.json"
    assert metadata_path.exists()

    with metadata_path.open() as f:
      metadata = json.load(f)

    assert metadata["sampler_type"] == "mcmc"
    assert metadata["git_commit_hash"] is None
    assert "timestamp" in metadata
    assert isinstance(metadata["config"], str)
    assert "TestConfig" in metadata["config"]

  def test_create_metadata_file_no_git(self, tmp_path: Path) -> None:
    """Test metadata file creation when git is unavailable.

    Args:
        tmp_path: Pytest fixture providing temporary directory.

    Returns:
        None

    Raises:
        AssertionError: If metadata file is not created correctly.

    Example:
        >>> test_create_metadata_file_no_git(tmp_path)

    """
    from dataclasses import dataclass

    @dataclass
    class TestConfig:
      sampler_type: str

    config = TestConfig(sampler_type="hmc")

    with patch("proteinsmc.io.get_git_commit_hash", return_value=None):
      create_metadata_file(config, tmp_path)

    metadata_path = tmp_path / "metadata.json"
    with metadata_path.open() as f:
      metadata = json.load(f)

    assert metadata["git_commit_hash"] is None


class TestCreateWriterCallback:
  """Test the create_writer_callback function and PyTree handling."""

  def test_create_writer_callback_returns_tuple(self, tmp_path: Path) -> None:
    """Test that create_writer_callback returns writer and callback.

    Args:
        tmp_path: Pytest fixture providing temporary directory.

    Returns:
        None

    Raises:
        AssertionError: If return types are incorrect.

    Example:
        >>> test_create_writer_callback_returns_tuple(tmp_path)

    """
    writer_path = str(tmp_path / "test_writer")
    writer, callback = create_writer_callback(writer_path)

    assert writer is not None
    assert callable(callback)
    writer.close()

  def test_writer_callback_with_flat_pytree(self, tmp_path: Path) -> None:
    """Test writer callback with a flat PyTree dictionary.

    Args:
        tmp_path: Pytest fixture providing temporary directory.

    Returns:
        None

    Raises:
        AssertionError: If writing or reading fails.

    Example:
        >>> test_writer_callback_with_flat_pytree(tmp_path)

    """
    writer_path = str(tmp_path / "test_flat_pytree")
    writer, callback = create_writer_callback(writer_path)

    # Create a flat PyTree payload as per spec: {"data": dict[str, PyTree]}
    payload = {
      "data": {
        "scalar": 42,
        "array": jnp.array([1, 2, 3]),
        "nested_scalar": 3.14,
      }
    }

    callback(payload)
    writer.close()

    # Verify the file was created
    from pathlib import Path

    assert Path(writer_path).exists()

  def test_writer_callback_with_nested_pytree(self, tmp_path: Path) -> None:
    """Test writer callback with nested PyTree structures.

    Args:
        tmp_path: Pytest fixture providing temporary directory.

    Returns:
        None

    Raises:
        AssertionError: If nested PyTree handling fails.

    Example:
        >>> test_writer_callback_with_nested_pytree(tmp_path)

    """
    writer_path = str(tmp_path / "test_nested_pytree")
    writer, callback = create_writer_callback(writer_path)

    # Create nested PyTree as per spec
    payload = {
      "data": {
        "level1": {
          "level2": {
            "array": jnp.array([[1, 2], [3, 4]]),
            "value": 100,
          },
          "simple": jnp.array([5, 6, 7]),
        },
        "top_level": jnp.array([8, 9]),
      }
    }

    callback(payload)
    writer.close()

    from pathlib import Path

    assert Path(writer_path).exists()

  def test_writer_callback_with_jax_arrays(self, tmp_path: Path) -> None:
    """Test writer callback specifically with JAX arrays.

    Args:
        tmp_path: Pytest fixture providing temporary directory.

    Returns:
        None

    Raises:
        AssertionError: If JAX array handling fails.

    Example:
        >>> test_writer_callback_with_jax_arrays(tmp_path)

    """
    writer_path = str(tmp_path / "test_jax_arrays")
    writer, callback = create_writer_callback(writer_path)

    # Test with various JAX array types
    payload = {
      "data": {
        "int_array": jnp.array([1, 2, 3], dtype=jnp.int32),
        "float_array": jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
        "bool_array": jnp.array([True, False, True], dtype=jnp.bool_),
        "multi_dim": jnp.ones((3, 4, 5)),
      }
    }

    callback(payload)
    writer.close()

    from pathlib import Path

    assert Path(writer_path).exists()

  def test_writer_callback_generality_with_jax_tree_util(
    self,
    tmp_path: Path,
  ) -> None:
    """Test that PyTree structure is handled generically using jax.tree_util.

    Args:
        tmp_path: Pytest fixture providing temporary directory.

    Returns:
        None

    Raises:
        AssertionError: If PyTree manipulation fails.

    Example:
        >>> test_writer_callback_generality_with_jax_tree_util(tmp_path)

    """
    writer_path = str(tmp_path / "test_tree_util")
    writer, callback = create_writer_callback(writer_path)

    # Create a complex PyTree and use jax.tree_util to inspect it
    payload = {
      "data": {
        "branch_a": {"leaf1": jnp.array([1, 2]), "leaf2": 42},
        "branch_b": jnp.array([3, 4, 5]),
      }
    }

    # Verify the tree structure before writing
    leaves, treedef = jtu.tree_flatten(payload)
    assert len(leaves) == 3  # Two arrays and one scalar
    reconstructed = jtu.tree_unflatten(treedef, leaves)
    assert reconstructed == payload

    callback(payload)
    writer.close()

    from pathlib import Path

    assert Path(writer_path).exists()

  def test_writer_callback_multiple_writes(self, tmp_path: Path) -> None:
    """Test multiple sequential writes to the same writer.

    Args:
        tmp_path: Pytest fixture providing temporary directory.

    Returns:
        None

    Raises:
        AssertionError: If multiple writes fail.

    Example:
        >>> test_writer_callback_multiple_writes(tmp_path)

    """
    writer_path = str(tmp_path / "test_multiple_writes")
    writer, callback = create_writer_callback(writer_path)

    # Write multiple payloads
    for i in range(5):
      payload = {
        "data": {
          "step": i,
          "values": jnp.array([i, i + 1, i + 2]),
        }
      }
      callback(payload)

    writer.close()

    from pathlib import Path

    assert Path(writer_path).exists()


class TestReadLineageData:
  """Test the read_lineage_data function."""

  def test_read_lineage_data_basic(self, tmp_path: Path) -> None:
    """Test reading basic lineage data.

    Args:
        tmp_path: Pytest fixture providing temporary directory.

    Returns:
        None

    Raises:
        AssertionError: If reading fails or data is incorrect.

    Example:
        >>> test_read_lineage_data_basic(tmp_path)

    """
    writer_path = str(tmp_path / "test_read_lineage")
    writer, callback = create_writer_callback(writer_path)

    # Write some test data
    test_records = [
      {"data": {"step": 0, "fitness": 1.0}},
      {"data": {"step": 1, "fitness": 1.5}},
      {"data": {"step": 2, "fitness": 2.0}},
    ]

    for record in test_records:
      callback(record)

    writer.close()

    # Read the data back
    lineage_data = read_lineage_data(writer_path)

    assert "records" in lineage_data
    assert len(lineage_data["records"]) == 3

  def test_read_lineage_data_with_arrays(self, tmp_path: Path) -> None:
    """Test reading lineage data containing JAX arrays.

    Args:
        tmp_path: Pytest fixture providing temporary directory.

    Returns:
        None

    Raises:
        AssertionError: If array data is not preserved.

    Example:
        >>> test_read_lineage_data_with_arrays(tmp_path)

    """
    writer_path = str(tmp_path / "test_read_arrays")
    writer, callback = create_writer_callback(writer_path)

    # Write data with arrays
    test_records = [
      {"data": {"sequences": jnp.array([[1, 2, 3], [4, 5, 6]])}},
      {"data": {"fitness_values": jnp.array([0.5, 0.8, 0.9])}},
    ]

    for record in test_records:
      callback(record)

    writer.close()

    # Read back and verify
    lineage_data = read_lineage_data(writer_path)

    assert len(lineage_data["records"]) == 2
    # Note: After deserialization, arrays become lists/numpy arrays
    assert "data" in lineage_data["records"][0]
    assert "sequences" in lineage_data["records"][0]["data"]

  def test_read_lineage_data_empty_file(self, tmp_path: Path) -> None:
    """Test reading from an empty lineage file.

    Args:
        tmp_path: Pytest fixture providing temporary directory.

    Returns:
        None

    Raises:
        AssertionError: If empty file handling fails.

    Example:
        >>> test_read_lineage_data_empty_file(tmp_path)

    """
    writer_path = str(tmp_path / "test_empty")
    writer, callback = create_writer_callback(writer_path)

    # Close without writing anything
    writer.close()

    # Read the empty file
    lineage_data = read_lineage_data(writer_path)

    assert "records" in lineage_data
    assert len(lineage_data["records"]) == 0

  def test_read_lineage_data_complex_pytree(self, tmp_path: Path) -> None:
    """Test reading lineage data with complex nested PyTree structures.

    Args:
        tmp_path: Pytest fixture providing temporary directory.

    Returns:
        None

    Raises:
        AssertionError: If complex PyTree is not preserved.

    Example:
        >>> test_read_lineage_data_complex_pytree(tmp_path)

    """
    writer_path = str(tmp_path / "test_complex_pytree")
    writer, callback = create_writer_callback(writer_path)

    # Write complex nested structure
    complex_payload = {
      "data": {
        "level1": {
          "level2": {"array": jnp.array([1, 2, 3]), "scalar": 42},
          "another_branch": jnp.array([[1, 2], [3, 4]]),
        },
        "top_value": 100,
      }
    }

    callback(complex_payload)
    writer.close()

    # Read and verify structure
    lineage_data = read_lineage_data(writer_path)

    assert len(lineage_data["records"]) == 1
    record = lineage_data["records"][0]
    assert "data" in record
    assert "level1" in record["data"]
    assert "level2" in record["data"]["level1"]
    assert "top_value" in record["data"]


class TestIntegrationScenarios:
  """Integration tests for complete I/O workflows."""

  def test_full_experiment_io_workflow(self, tmp_path: Path) -> None:
    """Test a complete experiment I/O workflow.

    Args:
        tmp_path: Pytest fixture providing temporary directory.

    Returns:
        None

    Raises:
        AssertionError: If workflow fails at any step.

    Example:
        >>> test_full_experiment_io_workflow(tmp_path)

    """
    from dataclasses import dataclass

    @dataclass
    class MockConfig:
      sampler_type: str
      num_steps: int
      mutation_rate: float

    config = MockConfig(sampler_type="smc", num_steps=10, mutation_rate=0.15)

    # Step 1: Create metadata
    with patch("proteinsmc.io.get_git_commit_hash", return_value="abc123"):
      create_metadata_file(config, tmp_path)

    # Step 2: Write lineage data
    writer_path = str(tmp_path / "lineage.array_record")
    writer, callback = create_writer_callback(writer_path)

    for step in range(5):
      payload = {
        "data": {
          "step": step,
          "sequences": jnp.array([[1, 2, 3], [4, 5, 6]]),
          "fitness": jnp.array([0.5 + step * 0.1, 0.6 + step * 0.1]),
        }
      }
      callback(payload)

    writer.close()

    # Step 3: Verify metadata
    metadata_path = tmp_path / "metadata.json"
    assert metadata_path.exists()

    with metadata_path.open() as f:
      metadata = json.load(f)
      assert metadata["sampler_type"] == "smc"
      assert metadata["git_commit_hash"] == "abc123"

    # Step 4: Read lineage data
    lineage_data = read_lineage_data(writer_path)
    assert len(lineage_data["records"]) == 5

  def test_pytree_roundtrip_preserves_structure(self, tmp_path: Path) -> None:
    """Test that PyTree structure is preserved through write/read cycle.

    Args:
        tmp_path: Pytest fixture providing temporary directory.

    Returns:
        None

    Raises:
        AssertionError: If PyTree structure is not preserved.

    Example:
        >>> test_pytree_roundtrip_preserves_structure(tmp_path)

    """
    writer_path = str(tmp_path / "roundtrip_test")
    writer, callback = create_writer_callback(writer_path)

    # Original PyTree structure
    original_payload = {
      "data": {
        "branch_a": {
          "leaf1": jnp.array([1.0, 2.0, 3.0]),
          "leaf2": 42,
        },
        "branch_b": jnp.array([[1, 2], [3, 4]]),
        "branch_c": {
          "nested": {"deep": jnp.array([5, 6, 7])},
        },
      }
    }

    # Get original structure
    original_leaves, original_treedef = jtu.tree_flatten(original_payload)

    # Write and read
    callback(original_payload)
    writer.close()

    lineage_data = read_lineage_data(writer_path)
    retrieved_payload = lineage_data["records"][0]

    # Verify structure is preserved (keys exist, nesting is maintained)
    assert "data" in retrieved_payload
    assert "branch_a" in retrieved_payload["data"]
    assert "leaf1" in retrieved_payload["data"]["branch_a"]
    assert "leaf2" in retrieved_payload["data"]["branch_a"]
    assert "branch_b" in retrieved_payload["data"]
    assert "branch_c" in retrieved_payload["data"]
    assert "nested" in retrieved_payload["data"]["branch_c"]
    assert "deep" in retrieved_payload["data"]["branch_c"]["nested"]
