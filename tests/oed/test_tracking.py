"""Tests for OED tracking functionality."""

import json
from pathlib import Path

import pytest

from proteinsmc.oed.structs import OEDDesign, OEDPredictedVariables
from proteinsmc.oed.tracking import (
  OEDRecordParams,
  add_oed_to_metadata,
  create_oed_summary,
  load_oed_checkpoint,
  load_oed_manifest,
  save_oed_checkpoint,
  save_oed_record,
)


@pytest.fixture
def sample_design() -> OEDDesign:
  """Create a sample OED design for testing."""
  return OEDDesign(
    N=20,
    K=2,
    q=4,
    population_size=100,
    mutation_rate=0.01,
    diversification_ratio=0.5,
    n_generations=10,
  )


@pytest.fixture
def sample_result() -> OEDPredictedVariables:
  """Create sample OED results for testing."""
  return OEDPredictedVariables(
    information_gain=1.5,
    barrier_crossing_frequency=0.3,
    final_sequence_entropy=2.1,
    jsd_from_original_population=0.4,
  )


def test_save_and_load_oed_record(
  tmp_path: Path, sample_design: OEDDesign, sample_result: OEDPredictedVariables
) -> None:
  """Test saving and loading OED records."""
  params = OEDRecordParams(
    output_dir=tmp_path,
    design=sample_design,
    result=sample_result,
    run_uuid="test-uuid-123",
    iteration=0,
    phase="seeding",
  )

  save_oed_record(params)

  # Verify manifest file was created
  manifest_file = tmp_path / "oed_manifest.jsonl"
  assert manifest_file.exists()

  # Load and verify
  records = load_oed_manifest(tmp_path)
  assert len(records) == 1
  assert records[0]["run_uuid"] == "test-uuid-123"
  assert records[0]["phase"] == "seeding"
  assert records[0]["iteration"] == 0


def test_save_multiple_records(
  tmp_path: Path, sample_design: OEDDesign, sample_result: OEDPredictedVariables
) -> None:
  """Test saving multiple records appends to manifest."""
  for i in range(3):
    params = OEDRecordParams(
      output_dir=tmp_path,
      design=sample_design,
      result=sample_result,
      run_uuid=f"uuid-{i}",
      iteration=i,
      phase="seeding",
    )
    save_oed_record(params)

  records = load_oed_manifest(tmp_path)
  assert len(records) == 3
  assert records[0]["run_uuid"] == "uuid-0"
  assert records[2]["run_uuid"] == "uuid-2"


def test_save_and_load_checkpoint(
  tmp_path: Path, sample_design: OEDDesign, sample_result: OEDPredictedVariables
) -> None:
  """Test checkpoint save and load."""
  design_history = [(sample_design, sample_result)]

  save_oed_checkpoint(tmp_path, design_history, current_iteration=5, seed=42)

  checkpoint = load_oed_checkpoint(tmp_path)
  assert checkpoint is not None
  assert checkpoint["current_iteration"] == 5
  assert checkpoint["seed"] == 42
  assert len(checkpoint["design_history"]) == 1  # type: ignore[arg-type]


def test_create_oed_summary(
  tmp_path: Path, sample_design: OEDDesign, sample_result: OEDPredictedVariables
) -> None:
  """Test OED summary creation."""
  # Create some records
  for i in range(5):
    result = OEDPredictedVariables(
      information_gain=float(i),
      barrier_crossing_frequency=0.1 * i,
      final_sequence_entropy=2.0 + i,
      jsd_from_original_population=0.3 + i * 0.1,
    )
    params = OEDRecordParams(
      output_dir=tmp_path,
      design=sample_design,
      result=result,
      run_uuid=f"uuid-{i}",
      iteration=i,
      phase="seeding" if i < 3 else "optimization",
    )
    save_oed_record(params)

  summary = create_oed_summary(tmp_path)

  assert summary["total_experiments"] == 5
  assert summary["seeding_experiments"] == 3
  assert summary["optimization_experiments"] == 2

  # Best info gain should be from the last experiment (i=4)
  best_info = summary["best_information_gain"]
  assert isinstance(best_info, dict)
  assert best_info["value"] == 4.0  # type: ignore[index]

  # Verify summary file was created
  summary_file = tmp_path / "oed_summary.json"
  assert summary_file.exists()


def test_add_oed_to_metadata(tmp_path: Path, sample_design: OEDDesign) -> None:
  """Test enhancing metadata with OED design info."""
  metadata_file = tmp_path / "metadata.json"

  # Create initial metadata
  initial_metadata = {
    "timestamp": "2025-01-01T00:00:00",
    "sampler_type": "smc",
  }
  with metadata_file.open("w") as f:
    json.dump(initial_metadata, f)

  # Add OED info
  add_oed_to_metadata(metadata_file, sample_design, phase="seeding", iteration=0)

  # Verify metadata was enhanced
  with metadata_file.open() as f:
    metadata = json.load(f)

  assert "oed_design" in metadata
  assert metadata["oed_phase"] == "seeding"
  assert metadata["oed_iteration"] == 0
  assert metadata["oed_design"]["N"] == 20
  assert metadata["oed_design"]["K"] == 2


def test_load_nonexistent_manifest(tmp_path: Path) -> None:
  """Test loading manifest when it doesn't exist."""
  records = load_oed_manifest(tmp_path)
  assert records == []


def test_load_nonexistent_checkpoint(tmp_path: Path) -> None:
  """Test loading checkpoint when it doesn't exist."""
  checkpoint = load_oed_checkpoint(tmp_path)
  assert checkpoint is None
