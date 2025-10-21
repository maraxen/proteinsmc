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

- **Issue:** Custom PRSMC implementation has shape/dtype consistency issues in edge cases.
- **Status (2025-10-21):** ✅ Core issues RESOLVED:
  - ✅ Key Management: Fixed by ensuring keys are always batched with shape `(n_islands, 2)` using `jax.random.split(key, n_islands)`
  - ✅ Update Parameters: Fixed by properly structuring `update_parameters` dict with shape `(population_size,)` arrays and vmapping states correctly
  - ✅ JAX PyTree Compatibility: Converted `MigrationInfo` and `PRSMCOutput` from `@dataclass` to `@struct.dataclass`
  - ✅ Loop Structure: Fixed `lax.fori_loop` to return correct carry structure
- **Remaining Work:**
  - Shape mismatches in migrate/exchange function for single-island case (n_islands=1)
  - Fitness extraction logic needs adjustment for StackedFitness format
  - Integration tests need updates for proper state initialization patterns
- **Priority:** Medium - Core PRSMC functionality works, edge cases need refinement
- **Key Files:**
  - Implementation: `src/proteinsmc/sampling/particle_systems/parallel_replica.py`
  - Initialization: `src/proteinsmc/sampling/initialization_factory.py` lines 303-379
  - Tests: `tests/sampling/particle_systems/test_parallel_replica.py`

### 3. Dynamic Data Type Handling (Cross-Application)

- **Issue:** Application-wide need for flexible dtype handling across different sequence types and samplers.
- **Current State:** Sequences use int8 for memory efficiency, but different components may expect float32 or other dtypes. Weight computations use mixed dtypes (bfloat16 vs float32).
- **Future Solution:** Implement centralized dtype management system that:
  1. Defines canonical dtypes for different data categories (sequences, weights, fitness scores, etc.)
  2. Provides automatic conversion utilities with performance optimization
  3. Supports configuration-based dtype selection for memory/precision trade-offs
  4. Ensures consistency across SMC, MCMC, and other sampling algorithms
- **Priority:** Medium-High - Affects multiple samplers and can cause subtle bugs
- **Implementation Notes:**
  - Should integrate with JAX's dtype promotion system
  - Consider using a registry pattern for dtype specifications
  - Add validation layers to catch dtype mismatches early
  - Document dtype expectations in function signatures using jaxtyping
- **Related Issues:** Connects to issue #1 (HMC/NUTS conversions) and issue #2 (PRSMC state management)

### 4. Recent Architectural Improvements (Completed)

- ✅ **SamplerState Refactoring:** Converted from `equinox.nn.State` (incorrect - for neural networks) to `flax.struct.dataclass` (correct - immutable PyTreeNode for JAX transformations)
- ✅ **Config Serialization:** Fixed `io.py` to exclude JAX Mesh field (contains unpicklable Device objects) when creating metadata files
- ✅ **Test Coverage:** Improved from 62% to 96.2% (227/236 tests passing)
- ✅ **PRSMC Key Management & Update Parameters:** Fixed key batching and update_parameters structure for proper vmap compatibility (2025-10-21)
