"""OED experiment tracking and persistence.

This module provides utilities for tracking OED experiments, including:
- Recording design-result pairs to a persistent manifest
- Linking OED designs to SMC run UUIDs
- Checkpointing and resuming OED loops
"""

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple, cast

from proteinsmc.oed.structs import OEDDesign, OEDPredictedVariables

logger = logging.getLogger(__name__)


class OEDRecordParams(NamedTuple):
  """Parameters for an OED record."""

  output_dir: str | Path
  design: OEDDesign
  result: OEDPredictedVariables
  run_uuid: str
  iteration: int
  phase: str = "seeding"
  record_start_idx: int = 0  # Starting index in the shared ArrayRecord
  record_count: int = 0  # Number of records written for this run


def _convert_jax_arrays(obj: object) -> object:
  """Recursively convert JAX arrays to Python primitives."""
  if hasattr(obj, "item"):  # JAX array or numpy scalar
    return obj.item()  # type: ignore[no-any-return]
  if isinstance(obj, dict):
    return {k: _convert_jax_arrays(v) for k, v in obj.items()}
  if isinstance(obj, list):
    return [_convert_jax_arrays(item) for item in obj]
  return obj


def save_oed_record(params: OEDRecordParams) -> None:
  """Append an OED experiment record to the manifest file.

  Args:
      params: OED record parameters containing design, result, run UUID, etc.

  """
  output_path = Path(params.output_dir)
  manifest_file = output_path / "oed_manifest.jsonl"

  # Convert dataclasses to dicts, handling JAX arrays
  design_dict = _convert_jax_arrays(asdict(params.design))
  result_dict = _convert_jax_arrays(asdict(params.result))

  record = {
    "timestamp": datetime.now(tz=UTC).isoformat(),
    "phase": params.phase,
    "iteration": params.iteration,
    "run_uuid": params.run_uuid,
    "design": design_dict,
    "result": result_dict,
    "record_start_idx": params.record_start_idx,
    "record_count": params.record_count,
  }

  # Append to JSONL file (one record per line)
  with manifest_file.open("a") as f:
    f.write(json.dumps(record) + "\n")

  logger.info(
    "Saved OED record: phase=%s, iteration=%d, run_uuid=%s, indices=[%d:%d]",
    params.phase,
    params.iteration,
    params.run_uuid,
    params.record_start_idx,
    params.record_start_idx + params.record_count,
  )


def load_oed_manifest(output_dir: str | Path) -> list[dict[str, object]]:
  """Load all OED records from the manifest file.

  Args:
      output_dir: Directory containing OED outputs

  Returns:
      List of OED records (dicts), in chronological order

  """
  output_path = Path(output_dir)
  manifest_file = output_path / "oed_manifest.jsonl"

  if not manifest_file.exists():
    logger.warning("No OED manifest found at %s", manifest_file)
    return []

  with manifest_file.open() as f:
    records = [json.loads(line) for line in f if line.strip()]

  logger.info("Loaded %d OED records from %s", len(records), manifest_file)
  return records


def get_next_record_index(output_dir: str | Path) -> int:
  """Get the next available record index in the shared ArrayRecord file.

  Args:
      output_dir: Directory containing OED outputs

  Returns:
      The starting index for the next run's records

  """
  records = load_oed_manifest(output_dir)
  if not records:
    return 0

  # Calculate the next available index from the last record
  last_record = records[-1]
  last_start_idx = last_record.get("record_start_idx", 0)
  last_count = last_record.get("record_count", 0)

  if isinstance(last_start_idx, int) and isinstance(last_count, int):
    return last_start_idx + last_count

  logger.warning("Invalid record indices in manifest, returning 0")
  return 0


def save_oed_checkpoint(
  output_dir: str | Path,
  design_history: list[tuple[OEDDesign, OEDPredictedVariables]],
  current_iteration: int,
  seed: int,
) -> None:
  """Save a checkpoint of the OED loop state.

  Args:
      output_dir: Directory containing OED outputs
      design_history: Current design-result history
      current_iteration: Current iteration number
      seed: Random seed for reproducibility

  """
  output_path = Path(output_dir)
  checkpoint_file = output_path / "oed_checkpoint.json"

  history_dicts = [
    {
      "design": _convert_jax_arrays(asdict(design)),
      "result": _convert_jax_arrays(asdict(result)),
    }
    for design, result in design_history
  ]

  checkpoint = {
    "timestamp": datetime.now(tz=UTC).isoformat(),
    "current_iteration": current_iteration,
    "seed": seed,
    "design_history": history_dicts,
  }

  with checkpoint_file.open("w") as f:
    json.dump(checkpoint, f, indent=2)

  logger.info(
    "Saved OED checkpoint at iteration %d with %d designs",
    current_iteration,
    len(design_history),
  )


def load_oed_checkpoint(output_dir: str | Path) -> dict[str, object] | None:
  """Load the most recent OED checkpoint.

  Args:
      output_dir: Directory containing OED outputs

  Returns:
      Checkpoint dict or None if no checkpoint exists

  """
  output_path = Path(output_dir)
  checkpoint_file = output_path / "oed_checkpoint.json"

  if not checkpoint_file.exists():
    logger.info("No checkpoint found at %s", checkpoint_file)
    return None

  with checkpoint_file.open() as f:
    checkpoint: dict[str, object] = json.load(f)

  logger.info(
    "Loaded checkpoint from iteration %d with %d designs",
    checkpoint["current_iteration"],
    len(checkpoint["design_history"]),  # type: ignore[arg-type]
  )
  return checkpoint


def create_oed_summary(output_dir: str | Path) -> dict[str, object]:
  """Create a summary of the OED experiment.

  Args:
      output_dir: Directory containing OED outputs

  Returns:
      Summary dict with statistics and best results

  """
  records = load_oed_manifest(output_dir)

  if not records:
    return {"error": "No OED records found"}

  # Calculate statistics
  seeding_records = [r for r in records if r["phase"] == "seeding"]
  optimization_records = [r for r in records if r["phase"] == "optimization"]

  # Find best designs by each metric
  def get_result_value(rec: dict[str, object], key: str) -> float:
    """Extract nested result value for comparison."""
    result = rec["result"]
    if isinstance(result, dict):
      result_dict = cast("dict[str, Any]", result)
      value = result_dict.get(key)
      if isinstance(value, (int, float)):
        return float(value)
    return float("-inf")

  best_info_gain = max(records, key=lambda r: get_result_value(r, "information_gain"))
  best_entropy = max(records, key=lambda r: get_result_value(r, "final_sequence_entropy"))
  best_jsd = max(records, key=lambda r: get_result_value(r, "jsd_from_original_population"))

  summary: dict[str, object] = {
    "total_experiments": len(records),
    "seeding_experiments": len(seeding_records),
    "optimization_experiments": len(optimization_records),
    "best_information_gain": {
      "value": get_result_value(best_info_gain, "information_gain"),
      "run_uuid": best_info_gain["run_uuid"],
      "design": best_info_gain["design"],
    },
    "best_final_entropy": {
      "value": get_result_value(best_entropy, "final_sequence_entropy"),
      "run_uuid": best_entropy["run_uuid"],
      "design": best_entropy["design"],
    },
    "best_jsd": {
      "value": get_result_value(best_jsd, "jsd_from_original_population"),
      "run_uuid": best_jsd["run_uuid"],
      "design": best_jsd["design"],
    },
    "first_experiment": records[0]["timestamp"],
    "last_experiment": records[-1]["timestamp"],
  }

  # Save summary to file
  output_path = Path(output_dir)
  summary_file = output_path / "oed_summary.json"
  with summary_file.open("w") as f:
    json.dump(summary, f, indent=2)

  logger.info("Created OED summary with %d total experiments", len(records))
  return summary


def add_oed_to_metadata(
  metadata_file: Path,
  design: OEDDesign,
  phase: str,
  iteration: int,
) -> None:
  """Add OED design information to an existing metadata file.

  Args:
      metadata_file: Path to the metadata.json file
      design: The OED design used for this run
      phase: Phase of OED loop ("seeding" or "optimization")
      iteration: Iteration number within the phase

  """
  if not metadata_file.exists():
    logger.warning("Metadata file not found: %s", metadata_file)
    return

  with metadata_file.open() as f:
    metadata: dict[str, object] = json.load(f)

  # Add OED information
  metadata["oed_design"] = _convert_jax_arrays(asdict(design))
  metadata["oed_phase"] = phase
  metadata["oed_iteration"] = iteration

  with metadata_file.open("w") as f:
    json.dump(metadata, f, indent=2)

  logger.info("Enhanced metadata with OED design: phase=%s, iteration=%d", phase, iteration)


def get_run_records_range(output_dir: str | Path, run_uuid: str) -> tuple[int, int] | None:
  """Get the record index range for a specific run.

  Args:
      output_dir: Directory containing OED outputs
      run_uuid: UUID of the run to find

  Returns:
      Tuple of (start_index, end_index) or None if run not found

  """
  records = load_oed_manifest(output_dir)

  for record in records:
    if record.get("run_uuid") == run_uuid:
      start_idx = record.get("record_start_idx", 0)
      count = record.get("record_count", 0)
      if isinstance(start_idx, int) and isinstance(count, int):
        return (start_idx, start_idx + count)

  logger.warning("Run UUID %s not found in manifest", run_uuid)
  return None


def get_shared_arrayrecord_path(output_dir: str | Path) -> Path:
  """Get the path to the shared ArrayRecord file for all OED runs.

  Args:
      output_dir: Directory containing OED outputs

  Returns:
      Path to the shared ArrayRecord file

  """
  return Path(output_dir) / "oed_data.arrayrecord"
