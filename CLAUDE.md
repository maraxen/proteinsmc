# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Dependencies
uv sync                                        # install all
uv sync --extra dev                            # include dev tools

# Tests
uv run pytest                                  # all tests
uv run pytest tests/models/test_smc.py        # single file
uv run pytest -k "test_name" -xvs             # single test, stop on failure

# Lint & type-check
uv run ruff check src/ --fix                  # lint (auto-fix enabled)
uv run ty check src/proteinsmc               # type checking

# Run an experiment
uv run python src/proteinsmc/oed/run.py       # OED outer loop
```

**Ruff config:** line-length=100, indent-width=2, `select = ["ALL"]` minus `PD`, `D203`, `D213`, `COM812`, `F722`, `TRY003`, `EM101/102`, `TC002`. Scripts in `scripts/` allow `T201` (print).

## Architecture

**proteinsmc** is a JAX-based Sequential Monte Carlo framework for protein sequence design and evolutionary studies. The two top-level goals are: (1) benchmarking samplers against each other on fitness landscapes, and (2) using the SMC framework to study protein evolution.

### Module map

| Module | Role |
|---|---|
| `runner.py` | Polymorphic entry point — routes to a sampler via `SAMPLER_REGISTRY` |
| `models/` | Frozen dataclass configs (`BaseSamplerConfig` → per-sampler subclasses) |
| `sampling/` | Sampler loop implementations (SMC via `particle_systems/`, plus Gibbs, MCMC, HMC, NUTS) |
| `scoring/` | Fitness functions: NK landscape, ProteinMPNN (`mpnn.py`), ESM2 (`esm.py`), Codon Adaptation Index (`cai.py`) |
| `oed/` | Optimal Experiment Design outer loop: GP models, acquisition, SMCTree batched runner |
| `io.py` | Centralized I/O — all outputs go through here |
| `types.py` | Global type aliases for sequences, arrays, PRNGKey |
| `utils/` | Constants, mutation helpers, memory tuning, metrics |

### Dispatch flow

`run_experiment(config)` in `runner.py` looks up `config.sampler_type` in `SAMPLER_REGISTRY` and calls the corresponding `run_fn`. Each sampler loop evolves a JAX PyTree state per generation, then passes output to I/O callbacks.

### All sampler state is a JAX PyTree

Mutable state is modeled as `flax.struct.dataclass` (registered PyTree) so JIT/vmap/scan work correctly. Never store state in Python-level mutable containers inside a sampler loop.

### Scoring functions are vmapped

Every scoring function in `scoring/` returns a function that takes a **batch** of sequences. They are built via factory functions (e.g. `make_nk_score()`, `make_mpnn_score()`) and vmapped over `population_size` automatically.

### OED outer loop (`oed/`)

Runs Bayesian optimization over sampler hyperparameters (N, K, q, population size, mutation rate). Uses GP models (`gp.py`) + a Fisher Information criterion to select the next design. `smc_tree.py` wraps batched SMC evaluation across candidate designs.

### I/O layout

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

## Key type conventions

```python
from jaxtyping import Array, Float, Int, PRNGKeyArray

# Sequences are int8, values 0–19 (protein) or 0–3 (nucleotide)
particles: Int[Array, "population_size sequence_length"]
weights:   Float[Array, "population_size"]
key:       PRNGKeyArray  # always split before use
```

## Adding a sampler

1. Config subclass in `models/<name>.py` extending `BaseSamplerConfig`
2. Loop function in `sampling/<name>.py` returning `SamplerOutput`
3. Register in `runner.py`: `SAMPLER_REGISTRY["name"] = {"config_cls": ..., "run_fn": ...}`
4. Tests in `tests/sampling/test_<name>.py`

## Known issues

- HMC/NUTS: 4 failing tests in `test_initialization_factory.py` due to int8→float32 type mismatch
- Parallel Replica SMC: edge cases remain for the single-island configuration
