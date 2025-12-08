"""Tests for I/O utilities including ArrayRecord operations and metadata generation."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from proteinsmc.io import (
  create_metadata_file,
  create_writer_callback,
  get_git_commit_hash,
  read_lineage_data,
)
from proteinsmc.models.sampler_base import SamplerOutput


class TestGetGitCommitHash:
  """Test the get_git_commit_hash function."""

  def test_get_git_commit_hash_success(self) -> None:
    """Test successful git commit hash retrieval."""
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
    """Test when git is not available."""
    with patch("shutil.which", return_value=None):
      commit_hash = get_git_commit_hash()

      assert commit_hash is None

  def test_get_git_commit_hash_not_in_repo(self) -> None:
    """Test when not in a git repository."""
    with patch("shutil.which", return_value="/usr/bin/git"):
      with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(128, ["git"])
        commit_hash = get_git_commit_hash()

        assert commit_hash is None

  def test_get_git_commit_hash_timeout(self) -> None:
    """Test when git command times out."""
    with patch("shutil.which", return_value="/usr/bin/git"):
      with patch("subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["git"], timeout=5)
        commit_hash = get_git_commit_hash()

        assert commit_hash is None


class TestCreateMetadataFile:
  """Test the create_metadata_file function."""

  def test_create_metadata_file_with_dataclass(self, tmp_path: Path) -> None:
    """Test metadata file creation with a dataclass config."""
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
    """Test metadata file creation with a non-dataclass object."""
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


class TestIO:
  """Test I/O operations with SamplerOutput and equinox."""

  def test_writer_callback_returns_tuple(self, tmp_path: Path) -> None:
    """Test that create_writer_callback returns writer and callback."""
    writer_path = str(tmp_path / "test_writer")
    writer, callback = create_writer_callback(writer_path)

    assert writer is not None
    assert callable(callback)
    writer.close()

  def test_write_and_read_single_step(self, tmp_path: Path) -> None:
    """Test writing and reading a single SamplerOutput step."""
    writer_path = str(tmp_path / "test_single")
    writer, callback = create_writer_callback(writer_path)

    # Create a dummy SamplerOutput
    output = SamplerOutput(
      step=jnp.array(0, dtype=jnp.int32),
      sequences=jnp.zeros((1, 5, 4), dtype=jnp.int32),
      fitness=jnp.zeros((1,), dtype=jnp.float32),
      key=jnp.zeros((2,), dtype=jnp.uint32),
      weights=jnp.ones((1,), dtype=jnp.float32)
    )

    callback(output)
    writer.close()

    # Read back
    skeleton = output
    data = list(read_lineage_data(writer_path, skeleton))

    assert len(data) == 1
    assert jnp.array_equal(data[0].step, output.step)
    assert jnp.array_equal(data[0].sequences, output.sequences)
    assert jnp.array_equal(data[0].weights, output.weights)

  def test_write_and_read_chunked(self, tmp_path: Path) -> None:
    """Test writing a chunk and reading it back as individual steps."""
    writer_path = str(tmp_path / "test_chunk")
    writer, callback = create_writer_callback(writer_path)

    # Create a single step output first to get all defaults
    single = SamplerOutput(
      step=jnp.array(0, dtype=jnp.int32),
      sequences=jnp.zeros((1, 5, 4), dtype=jnp.int32),
      fitness=jnp.zeros((1,), dtype=jnp.float32),
      key=jnp.zeros((2,), dtype=jnp.uint32),
      weights=jnp.ones((1,), dtype=jnp.float32)
    )

    # Broadcast to create chunk of size 2
    # This ensures ALL fields (even defaults) are stacked
    chunked_output = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), single)

    # Update step to be [0, 1]
    chunked_output = eqx.tree_at(
        lambda t: t.step,
        chunked_output,
        jnp.array([0, 1], dtype=jnp.int32)
    )

    callback(chunked_output)
    writer.close()

    # Skeleton should be single step
    skeleton = single

    data = list(read_lineage_data(writer_path, skeleton))

    assert len(data) == 2
    assert data[0].step == 0
    assert data[1].step == 1
    assert data[0].sequences.shape == (1, 5, 4)

  def test_read_empty_file(self, tmp_path: Path) -> None:
    """Test reading from an empty file."""
    writer_path = str(tmp_path / "test_empty")
    writer, _ = create_writer_callback(writer_path)
    writer.close()

    skeleton = SamplerOutput(
      step=jnp.array(0),
      sequences=jnp.zeros((1, 5, 4)),
      fitness=jnp.zeros((1,)),
      key=jnp.zeros((2,), dtype=jnp.uint32)
    )

    data = list(read_lineage_data(writer_path, skeleton))
    assert len(data) == 0

