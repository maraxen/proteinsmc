# OED Tracking System

## Overview

The OED (Optimal Experimental Design) tracking system provides comprehensive persistence and analysis capabilities for OED experiments. It tracks all design-result pairs, links them to SMC run UUIDs, and supports checkpointing and resumption of OED loops.

## Features

### 1. **Experiment Manifest** (`oed_manifest.jsonl`)
- JSONL format (one record per line) for append-only tracking
- Each record contains:
  - `timestamp`: When the experiment was run
  - `phase`: "seeding" or "optimization"
  - `iteration`: Iteration number within the phase
  - `run_uuid`: UUID of the corresponding SMC run
  - `design`: OED design parameters (N, K, q, population_size, etc.)
  - `result`: Predicted variables (information_gain, entropy, etc.)

### 2. **Enhanced Metadata**
- Each SMC run's `metadata.json` is enhanced with:
  - `oed_design`: The OED design used for this run
  - `oed_phase`: Phase of the OED loop
  - `oed_iteration`: Iteration number

### 3. **Checkpointing** (`oed_checkpoint.json`)
- Periodic checkpoints of the OED loop state
- Contains:
  - `current_iteration`: Progress tracking
  - `seed`: Random seed for reproducibility
  - `design_history`: All design-result pairs so far
- Enables resuming interrupted OED loops

### 4. **Summary Generation** (`oed_summary.json`)
- Aggregated statistics across all experiments
- Identifies best designs by each metric:
  - Best information gain
  - Best final entropy
  - Best JSD from original population
- Includes phase breakdown (seeding vs. optimization)

## Usage

### Running OED with Tracking

The tracking is automatically integrated into the OED loop in `src/proteinsmc/oed/run.py`:

```python
python -m src.proteinsmc.oed.run --output_dir oed_results --num_initial_experiments 20 --num_oed_iterations 100
```

This will create:
- `oed_results/oed_manifest.jsonl`: All experiment records
- `oed_results/oed_checkpoint.json`: Latest checkpoint
- `oed_results/data_<uuid>.arrayrecord`: SMC outputs
- `oed_results/metadata.json`: Enhanced with OED info (note: this gets overwritten, use manifest for history)

### Viewing Results

Use the provided CLI tool:

```bash
# Show basic stats
./scripts/view_oed_tracking.py oed_results

# Generate and view summary
./scripts/view_oed_tracking.py oed_results --summary

# List all experiments
./scripts/view_oed_tracking.py oed_results --list
```

### Programmatic Access

```python
from pathlib import Path
from proteinsmc.oed.tracking import (
    load_oed_manifest,
    load_oed_checkpoint,
    create_oed_summary,
)

# Load all experiment records
records = load_oed_manifest(Path("oed_results"))

# Load latest checkpoint
checkpoint = load_oed_checkpoint(Path("oed_results"))

# Generate summary
summary = create_oed_summary(Path("oed_results"))
```

### Manual Tracking

If running individual OED experiments:

```python
from proteinsmc.oed.tracking import OEDRecordParams, save_oed_record

# After running an experiment
save_oed_record(
    OEDRecordParams(
        output_dir=output_dir,
        design=design,
        result=result,
        run_uuid=run_uuid,
        iteration=i,
        phase="seeding",
    )
)
```

## File Structure

```
oed_results/
├── oed_manifest.jsonl          # All experiment records (append-only)
├── oed_checkpoint.json         # Latest checkpoint
├── oed_summary.json            # Generated summary
├── oed_loop.log               # Execution log
├── metadata.json              # Most recent SMC run metadata (enhanced)
├── data_<uuid1>.arrayrecord   # SMC run 1 output
├── data_<uuid2>.arrayrecord   # SMC run 2 output
└── ...
```

## Data Linkage

The system maintains a complete chain of linkage:

1. **OED Design** → **Run UUID** (via `oed_manifest.jsonl`)
2. **Run UUID** → **SMC Output** (via `data_<uuid>.arrayrecord`)
3. **SMC Output** → **Results** (via deserialization)
4. **Results** → **Metrics** (information gain, entropy, etc.)

This allows you to:
- Find which design produced which results
- Trace results back to the original SMC trajectory
- Analyze the full sampling history for any experiment
- Resume OED loops from checkpoints

## API Reference

### `save_oed_record(params: OEDRecordParams)`
Save an OED experiment record to the manifest.

### `load_oed_manifest(output_dir: Path) -> list[dict]`
Load all OED records from the manifest file.

### `save_oed_checkpoint(output_dir, design_history, current_iteration, seed)`
Save a checkpoint of the OED loop state.

### `load_oed_checkpoint(output_dir: Path) -> dict | None`
Load the most recent OED checkpoint.

### `create_oed_summary(output_dir: Path) -> dict`
Generate a summary of all OED experiments.

### `add_oed_to_metadata(metadata_file, design, phase, iteration)`
Enhance an SMC run's metadata with OED design information.

## Best Practices

1. **Always use the manifest** for historical queries (not `metadata.json`, which gets overwritten)
2. **Checkpoint frequently** to enable resumption after failures
3. **Generate summaries** periodically to monitor progress
4. **Keep UUIDs** to trace results back to specific runs
5. **Use the CLI tool** for quick inspection and debugging

## Future Enhancements

Potential additions (see `AGENTS.md` Technical Debt section):
- Resume OED loop from checkpoint
- Visualization tools for design space exploration
- Integration with experiment tracking platforms (e.g., Weights & Biases)
- Automated hyperparameter tuning based on tracking data
