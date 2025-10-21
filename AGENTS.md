# GEMINI.md - Development Guidelines for Protein SMC Experiment

This document provides guidelines and context for continuing development on the Protein SMC Experiment project.

## Project Goals

The primary goals for this repository are:

1. **Benchmarking:** Evaluate the performance of the JAX-based Sequential Monte Carlo (SMC) sampling approach against other protein sequence sampling methods. This could involve comparing convergence rates, diversity of generated sequences, and computational efficiency.
2. **Evolutionary Studies:** Utilize the SMC framework to study protein evolution, including exploring fitness landscapes, identifying evolutionary pathways, and understanding the impact of different selective pressures.

## Current Status and Future Work

### Island Models

A sophisticated island model is implemented in the form of a **Parallel Replica SMC algorithm** located in `src/sampling/parallel_replica.py`. This implementation addresses the core concepts of island models for evolutionary studies and benchmarking.

The key features of this implementation are:

- **Multiple Islands:** The `run_parallel_replica_smc_jax` function runs multiple independent SMC simulations in parallel, referred to as "islands." Each island maintains its own population of particles.
- **Replica Exchange:** The model includes a replica exchange strategy where configurations (particles) are swapped between islands based on a Metropolis-Hastings-like criterion. This allows for information to be shared between different temperature replicas, improving sampling efficiency.
- **State Management:** The `IslandState` NamedTuple is used to manage the state of each island, including the particles, random keys, and other parameters.
- **Flexible Configuration:** The implementation allows for configuring the number of islands, the number of particles per island, the exchange frequency, and other parameters.

### Benchmarking

To benchmark this SMC implementation, consider:

- **Defining Metrics:** What constitutes "good" sampling? (e.g., coverage of sequence space, proximity to optimal sequences, diversity).
- **Comparison Targets:** Identify other sampling algorithms or tools (e.g., MCMC, genetic algorithms, other protein design software) to compare against.
- **Test Cases:** Create specific protein design problems or fitness landscapes to test the algorithms.

### Evolutionary Studies

For evolutionary studies, you might want to:

- **Fitness Functions:** Experiment with different fitness functions that represent various evolutionary pressures (e.g., stability, binding affinity, catalytic activity).
- **Landscape Analysis:** Use the sampled sequences to map and analyze protein fitness landscapes.
- **Ancestral Reconstruction:** Potentially adapt the SMC framework for ancestral sequence reconstruction or inferring evolutionary histories.

## Development Practices

### Testing

While there's a `tests/` directory, ensure that new features and modifications are accompanied by appropriate unit and integration tests. This is crucial for maintaining code quality and verifying the correctness of complex simulations.

### Linting

The project uses `ruff` for linting. The current configuration focuses on `F` (Pyflakes) and `B` (Bugbear) error codes, and ignores `E501` (line length). Always run `ruff check src/ --fix` before committing changes to ensure code quality and consistency.

### Code Structure

Maintain the modular structure within `src/`. When adding new functionalities, consider where they best fit within the existing `experiment.py`, `initiation.py`, `mpnn.py`, `mutate.py`, `sampling/`, and `utils/` modules, or if a new module is warranted.

### Running Python Commands

This project uses `uv` for dependency and environment management. Always use `uv run` to execute Python commands, which automatically handles the virtual environment.

### Running Tests

To run all tests, use the following command from the project root:

```bash
uv run pytest
```

To run tests with coverage:

```bash
uv run pytest --cov=src --cov-report=term
```

To run a specific test file, provide its path:

```bash
uv run pytest tests/utils/test_combined_fitness.py
```

### Shell Commands

When using shell commands, especially when dealing with paths that might contain spaces or special characters, always enclose the paths in quotes.

### Dependencies

Keep `requirements.txt` up-to-date with any new Python package dependencies.

By following these guidelines, we can ensure consistent, high-quality development for the Protein SMC Experiment project.

## LAST SESSION

**Instructions for Models:**
Review this section at the beginning of each session to understand the current state of work. Update this section upon completing a significant milestone or when handing off the session.

**Work Accomplished:**
-

**Remaining Work:**

- Scalable Experiment Management and I/O

**Challenges Encountered:**
-

# **Additional Gemini Development Guide for proteinsmc added 250717**

Thproteis document outlines the high-level goals, coding style, and architectural decisions for the proteinsmc repository. Please use this as your primary guide for all development and refactoring tasks.

## **Core Architectural Strategy**

Based on recent planning, the project has adopted a new, scalable architecture for running experiments and managing data. All new development should align with these principles:

1. **I/O and Data Pipeline (Hot \-\> Warm \-\> Cold):**  
   - **Core Module:** All I/O operations are managed by src/proteinsmc/io.py, which contains the RunManager and asynchronous DataWriter classes. The old parquet\_io.py is deprecated.  
   - **Run Identification:** Each experiment is identified by a uuid7.  
   - **Storage (Hot):** During a run, data is written to a flat output directory. Tensor data is buffered and saved in batches to .safetensors files. Scalar metrics are appended to a .jsonl file. This avoids creating thousands of small files or directories per run.  
   - **Storage (Warm/Cold):** Post-processing scripts located in the scripts/ directory are used to consolidate run data and create master tables (e.g., Parquet) for efficient querying.  
2. **Experiment Runner:**  
   - **Polymorphic Dispatch:** The main entry point for all experiments is the run\_experiment function in src/proteinsmc/runner.py. This function uses the sampler\_type field in the configuration object to dispatch to the correct sampler logic.  
3. **JAX Data Structures:**  
   - **Static vs. Dynamic:** We distinguish between static configuration and dynamic state.  
   - **Configuration (static):** All sampler configurations (e.g., SMCConfig) inherit from BaseSamplerConfig and contain only static parameters.  
   - **State (dynamic):** The dynamic state that changes at every step of a sampler (e.g., SMCCarryState) must be a flax.struct.PyTreeNode to ensure it is handled correctly by jax.jit and jax.lax.scan.

## **High-Level Goals**

1. **Implement the Core Architectural Strategy:** The immediate priority is to fully implement the refactoring plan. This involves creating the new io.py and runner.py modules, refactoring the data classes, and updating the samplers to use the new system.  
2. **Expand Sampler Support:** Add more sampler types (e.g., Parallel Replica) to the polymorphic run\_experiment dispatcher.  
3. **Enhance Post-Processing:** Develop robust scripts in the scripts/ directory for analyzing and visualizing results from the new data format.  
4. **Continuous Testing:** Ensure all new components are thoroughly tested. I/O logic should be tested separately from the core computational logic of the samplers.

## **Coding Style and Best Practices**

- **Type Hinting:** All new code must be fully type-hinted. Use jaxtyping for JAX arrays.  
- **Docstrings:** Provide clear and concise docstrings for all modules, classes, and functions.  
- **Testing:** Use pytest for testing. Use chex for assertions on JAX arrays. Mock external systems like file I/O where appropriate.  
- **Immutability:** Embrace the functional and immutable nature of JAX. Avoid in-place modifications of state.  
- **Clarity over Premature Optimization:** Write clear, readable code first. Rely on jax.jit for performance and only optimize further if a bottleneck is identified.

## **Technical Debt & Future Enhancements Backlog**

### 1. Data Type Conversions (HMC/NUTS)

- **Issue:** HMC/NUTS samplers require float32 inputs for gradient computation, but sequences are stored as int8 for memory efficiency.
- **Current State:** 4 tests failing in `test_initialization_factory.py` (HMC/NUTS initialization with int8 sequences).
- **Future Solution:** Implement modular data type conversion layer that:
  1. Detects sampler requirements (gradient-based vs. discrete)
  2. Automatically converts int8 sequences to float32 when needed
  3. Converts results back to int8 for storage
  4. Minimizes memory overhead through lazy conversion
- **Priority:** Medium - Currently blocking HMC/NUTS testing, but these samplers are less critical than SMC for primary use cases.
- **Implementation Notes:**
  - Consider using JAX's `astype()` with JIT compilation for efficiency
  - May need to update `StackedFitnessFn` type signature to be more flexible
  - Document the performance trade-offs in the conversion layer
- **Test Files:** `tests/sampling/test_initialization_factory.py` lines 96-179

### 2. Parallel Replica SMC State Management

- **Issue:** Custom PRSMC implementation has architectural mismatches with BlackJAX SMC API around state batching.
- **Current State:** 3-4 tests failing in `test_parallel_replica.py::TestRunPRSMCLoop` with various errors:
  - `IndexError: tuple index out of range` in basic loop test
  - `TypeError: JAX encountered invalid PRNG key data` in no-exchange tests
  - Complex interaction between per-island state and BlackJAX SMC expectations
- **Root Causes:**
  1. **Key Management:** PRSMC uses per-island keys (shape `(n_islands, 2)`), but key splitting logic may not handle this correctly in all code paths
  2. **Update Parameters Structure:** BlackJAX SMC's `update_parameters` field needs to be a dict with batched values for custom update functions, but the batching semantics across islands are unclear
  3. **State Batching:** Unclear whether BlackJAX SMC expects states to be batched via vmap or handled differently for multi-replica scenarios
- **Key Files:**
  - Implementation: `src/proteinsmc/sampling/particle_systems/parallel_replica.py`
  - Initialization: `src/proteinsmc/sampling/initialization_factory.py` lines 303-379 (`_initialize_prsmc_state`)
  - Tests: `tests/sampling/particle_systems/test_parallel_replica.py` lines 514-805
- **Priority:** Medium-High - Parallel replica is a core feature for evolutionary studies and benchmarking.
- **Next Steps:**
  1. Consult BlackJAX documentation on proper SMC state structure for custom update functions
  2. Clarify whether `vmap` over BlackJAX SMC states is the correct pattern or if there's a better way
  3. Review how `update_parameters` dict should be structured when vmapped
  4. Investigate the `IndexError: tuple index out of range` - likely related to key indexing after vmap splits
  5. Consider whether parallel replica should use a different BlackJAX primitive or custom implementation
- **Context:** BlackJAX does not have native support for parallel replica exchange - this is a custom wrapper around BlackJAX SMC.

### 3. Recent Architectural Improvements (Completed)

- ✅ **SamplerState Refactoring:** Converted from `equinox.nn.State` (incorrect - for neural networks) to `flax.struct.dataclass` (correct - immutable PyTreeNode for JAX transformations)
- ✅ **Config Serialization:** Fixed `io.py` to exclude JAX Mesh field (contains unpicklable Device objects) when creating metadata files
- ✅ **Test Coverage:** Improved from 62% to 96.2% (227/236 tests passing)
