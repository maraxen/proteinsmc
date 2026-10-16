# proteinsmc

## Project Goals

The primary goals for this repository are:

1. **Benchmarking:** Evaluate the performance of the JAX-based Sequential Monte Carlo (SMC) sampling approach against other protein sequence sampling methods. This could involve comparing convergence rates, diversity of generated sequences, and computational efficiency.
2. **Evolutionary Studies:** Utilize the SMC framework to study protein evolution, including exploring fitness landscapes, identifying evolutionary pathways, and understanding the impact of different selective pressures.

## Commands

```bash
# Dependencies
uv sync                                        # install all
uv sync --extra dev                            # include dev tools

# Tests
uv run pytest                                  # all tests
uv run pytest tests/models/test_smc.py        # single file
uv run pytest -k "test_name" -xvs             # single test, stop on failure
uv run pytest --cov=src --cov-report=term

# Lint & type-check
uv run ruff check src/ --fix                  # lint (auto-fix enabled)
uv run ty check src/proteinsmc               # type checking

# Run an experiment
uv run python src/proteinsmc/oed/run.py       # OED outer loop
```

**Ruff config:** line-length=100, indent-width=2, `select = ["ALL"]` minus `PD`, `D203`, `D213`, `COM812`, `F722`, `TRY003`, `EM101/102`, `TC002`. Scripts in `scripts/` allow `T201` (print).

### Shell Commands

When using shell commands, especially when dealing with paths that might contain spaces or special characters, always enclose the paths in quotes.

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

## Core Architectural Strategy

Based on recent planning, the project has adopted a new, scalable architecture for running experiments and managing data. All new development should align with these principles:

3. **JAX Data Structures:**  
   - **Static vs. Dynamic:** We distinguish between static configuration and dynamic state.  
   - **Configuration (static):** All sampler configurations (e.g., SMCConfig) inherit from BaseSamplerConfig and contain only static parameters.  

## **Coding Style and Best Practices**

- **Type Hinting:** All new code must be fully type-hinted. Use jaxtyping for JAX arrays.  
- **Docstrings:** Provide clear and concise docstrings for all modules, classes, and functions.  
- **Testing:** Use pytest for testing. Use chex for assertions on JAX arrays. Mock external systems like file I/O where appropriate.  
- **Immutability:** Embrace the functional and immutable nature of JAX. Avoid in-place modifications of state.  
- **Clarity over Premature Optimization:** Write clear, readable code first. Rely on jax.jit for performance and only optimize further if a bottleneck is identified.

## Pointers

- **Writing or reading run outputs, or editing `io.py`** → `.praxia/docs/reference/261016_run-output-io.md`
