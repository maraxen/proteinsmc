---
title: 'Run output I/O layout and strategy'
description: Run-output directory layout and the hot/warm/cold I/O strategy, moved verbatim out of the always-on harness files; unreconciled with src/proteinsmc/io.py.
status: active
task_id: 261016_harness-fleet-audit
date: '261016'
---

Verification note (261016): `src/proteinsmc/io.py` defines `get_git_commit_hash`, `create_metadata_file`, `create_writer_callback` (an `ArrayRecordWriter`), `read_lineage_data` and `read_lineage_data_range`. There is no `RunManager` or `DataWriter` class, and no `safetensors`, `metrics.jsonl` or `uuid7` string under `src/`. Treat the text below as design intent until reconciled with the code.

### I/O layout (from CLAUDE.md)

Each run gets a UUID directory under `outputs/`:
```
outputs/<run_uuid>/
  metadata.json          # config, git SHA, timing
  particles.safetensors  # batched sequences
  weights.safetensors
  fitness_scores.safetensors
  metrics.jsonl          # one record per generation (append-only)
  oed_results.jsonl      # OED-specific (if applicable)
```

Tensors use `.safetensors`; scalars use `.jsonl`. No per-generation subdirectories.

### I/O and Data Pipeline (from AGENTS.md "Core Architectural Strategy", item 1)

1. **I/O and Data Pipeline (Hot \-\> Warm \-\> Cold):**  
   - **Core Module:** All I/O operations are managed by src/proteinsmc/io.py, which contains the RunManager and asynchronous DataWriter classes. The old parquet\_io.py is deprecated.  
   - **Run Identification:** Each experiment is identified by a uuid7.  
   - **Storage (Hot):** During a run, data is written to a flat output directory. Tensor data is buffered and saved in batches to .safetensors files. Scalar metrics are appended to a .jsonl file. This avoids creating thousands of small files or directories per run.  
   - **Storage (Warm/Cold):** Post-processing scripts located in the scripts/ directory are used to consolidate run data and create master tables (e.g., Parquet) for efficient querying.  
