# **Refactoring Plan: Scalable Experiment Management and I/O**

**Disclaimer:** This document provides a strategic guide based on the state of the repository at the time of its creation. As the codebase evolves, some specific class names, method signatures, or file structures mentioned herein may change. Always refer to the current state of the repository for the most accurate implementation details.

**Tracking:** Steps marked with `[DONE]` are complete.

This document provides a comprehensive strategy for refactoring the proteinsmc repository. The goal is to implement a robust and scalable data pipeline capable of handling tens of thousands of experiments, while simultaneously improving the design of core data structures and creating a polymorphic system for running various sampler types.

### Table of Contents

1.  **High-Level Strategy: The Hot-Warm-Cold Pipeline**
2.  **Phase 1: A New, Scalable I/O System**
3.  **Phase 2: Simplifying the Core Models (Fitness, Annealing, and State)**
4.  **Phase 3: Integrating the New System**
5.  **Phase 4: Testing the New Implementation**
6.  **Phase 5: Advanced Considerations & Future-Proofing**

### 1\. High-Level Strategy: The Hot-Warm-Cold Pipeline

The current approach of creating one directory per run is not scalable. We will implement a Hot -> Warm -> Cold data pipeline that minimizes file operations during a run and consolidates data for efficient long-term storage and analysis.

*   **🔥 HOT (During the Run):** Tensor data is batched to `.safetensors` files; scalar metrics are appended to a `.jsonl` file.
*   **☀️ WARM (Post-Run Consolidation):** A script consolidates "hot" files.
*   **🧊 COLD (Long-Term Archival):** A final script transforms "warm" data into a query-efficient **Parquet** file.

---

### Phase 1: A New, Scalable I/O System

This phase replaces the logic in `src/proteinsmc/utils/parquet_io.py` with a new, more robust module.

*   `[DONE]` **Step 1.1: Add New Dependencies.** Add `uuid-utils`, `safetensors`, `safejax` to `requirements.txt`.
*   `[DONE]` **Step 1.2: Create the RunManager and DataWriter.** Create `src/proteinsmc/io.py` with non-blocking I/O using a `ThreadPoolExecutor`.
*   `[DONE]` **Step 1.3: Define the New Directory Structure.** All output files will now live in a single, flat directory.
*   `[DONE]` **Step 1.4: Implement Post-Processing Scripts.** Create a `scripts/` directory at the root of the repository.

---

### Phase 2: Simplifying the Core Models (Fitness, Annealing, and State)

The core of this phase is to **stop misusing JAX PyTrees for static configuration**. We will refactor all configuration objects to be simple dataclasses and eliminate the complex `Registry` system entirely. We will use `jax.jit`'s `static_argnames` feature to pass function objects directly to the JIT-compiled sampler, which is more efficient and idiomatic.

*   `[DONE]` **Step 2.1: Refactor `BaseSamplerConfig`.**
    *   **File:** `src/proteinsmc/models/sampler_base.py`
    *   **Action:** Remove PyTree registration and `tree_flatten`/`tree_unflatten` methods. Add a `sampler_type: str` field to enable polymorphic dispatch.

*   `[DONE]` **Step 2.2: Refactor `SMCConfig` and `SMCCarryState`.**
    *   **File:** `src/proteinsmc/models/smc.py`
    *   **Action:**
        *   In `SMCConfig`, remove all PyTree logic and add `sampler_type: str = field(default="smc", init=False)`.
        *   Change `SMCCarryState` to be a `flax.struct.PyTreeNode` and remove its manual PyTree methods. This clearly separates static config from dynamic state.

*   `[DONE]` **Step 2.3: Delete the Entire Registry System.**
    *   **Action:** This system is overly complex and unnecessary with the new design. Delete the following files:
        *   `src/proteinsmc/models/registry_base.py`
        *   `src/proteinsmc/scoring/registry.py`

*   `[DONE]` **Step 2.4: Simplify `AnnealingScheduleConfig`.**
    *   **File:** `src/proteinsmc/models/annealing.py`
    *   **Action:** Remove all `Registry` and `RegisteredFunction` logic. Make `AnnealingScheduleConfig` a simple, non-PyTree dataclass. The `__call__` method will be removed.
    *   **File:** `src/proteinsmc/utils/annealing.py`
    *   **Action:** Remove the `ANNEALING_REGISTRY`. The file will now just contain the schedule functions (e.g., `linear_schedule`).

*   `[DONE]` **Step 2.5: Simplify `FitnessEvaluator`.**
    *   **File:** `src/proteinsmc/models/fitness.py`
    *   **Action:** Remove all `Registry` and `RegisteredFunction` logic. `FitnessEvaluator` and `FitnessFunction` will become simple, non-PyTree dataclasses.
    *   **File:** `src/proteinsmc/scoring/cai.py`, `mpnn.py`, `combine.py`
    *   **Action:** Remove the `FitnessRegistryItem` and `CombineRegistryItem` wrappers. The `make_*` functions will now return the raw, JIT-compatible functions.

---

### Phase 3: Integrating the New System

This phase focuses on creating a generic, registry-based runner that can dispatch to different samplers based on the configuration.

*   `[DONE]` **Step 3.1: Create a Generic Sampler Dispatcher.**
    *   **File:** `src/proteinsmc/runner.py`
    *   **Action:** Created the main `run_experiment` entry point. This function uses a `SAMPLER_REGISTRY` dictionary to:
        1.  Look up the correct sampler functions (`initialize_fn`, `run_fn`) based on `config.sampler_type`.
        2.  Use factory functions (`get_fitness_function`, `get_annealing_function`) to create JIT-compatible functions from the configuration.
        3.  Call the core JIT-compiled sampler, passing the function objects as arguments.
        4.  Loop through the results and use the `DataWriter` to save them.

*   `[DONE]` **Step 3.2: Formalize Initialization Logic.**
    *   **File:** `src/proteinsmc/utils/initiate.py`
    *   **Action:** Refactored this file to contain a clear `initialize_smc_state(config, key)` function, which is now used by the generic runner.


---

### Phase 4: Testing the New Implementation

*   `[ ]` **Step 4.1: Testing the I/O Module.**
    *   **File:** `tests/test_io.py` (new)
    *   **Action:** Create a test to verify that the `RunManager` and `DataWriter` write files correctly, especially handling the asynchronous writes.

*   `[ ]` **Step 4.2: Updating and Refactoring Existing Tests.**
    *   **Action:** Update test fixtures to use the new, simplified configuration objects. Mock the `RunManager` and `DataWriter` to test sampler logic in isolation. Focus tests on the logic of the core JIT-compiled functions.

---

### Phase 5: Advanced Considerations & Future-Proofing

*   **Truly Non-Blocking I/O:** The `ThreadPoolExecutor` in the `DataWriter` is a crucial enhancement that prevents the GPU from sitting idle while waiting for file I/O.
*   **Configuration Management at Scale:** For larger projects, consider tools like **Hydra** or **Gin-config**. The new design, which separates configuration from execution, makes integrating such tools much easier in the future.