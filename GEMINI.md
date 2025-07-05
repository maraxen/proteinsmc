# GEMINI.md - Development Guidelines for Protein SMC Experiment

This document provides guidelines and context for continuing development on the Protein SMC Experiment project.

## Project Goals
The primary goals for this repository are:
1.  **Benchmarking:** Evaluate the performance of the JAX-based Sequential Monte Carlo (SMC) sampling approach against other protein sequence sampling methods. This could involve comparing convergence rates, diversity of generated sequences, and computational efficiency.
2.  **Evolutionary Studies:** Utilize the SMC framework to study protein evolution, including exploring fitness landscapes, identifying evolutionary pathways, and understanding the impact of different selective pressures.

## Current Status and Future Work

### Island Models
A sophisticated island model is implemented in the form of a **Parallel Replica SMC algorithm** located in `src/sampling/parallel_replica.py`. This implementation addresses the core concepts of island models for evolutionary studies and benchmarking.

The key features of this implementation are:
-   **Multiple Islands:** The `run_parallel_replica_smc_jax` function runs multiple independent SMC simulations in parallel, referred to as "islands." Each island maintains its own population of particles.
-   **Replica Exchange:** The model includes a replica exchange strategy where configurations (particles) are swapped between islands based on a Metropolis-Hastings-like criterion. This allows for information to be shared between different temperature replicas, improving sampling efficiency.
-   **State Management:** The `IslandState` NamedTuple is used to manage the state of each island, including the particles, random keys, and other parameters.
-   **Flexible Configuration:** The implementation allows for configuring the number of islands, the number of particles per island, the exchange frequency, and other parameters.

### Benchmarking
To benchmark this SMC implementation, consider:
-   **Defining Metrics:** What constitutes "good" sampling? (e.g., coverage of sequence space, proximity to optimal sequences, diversity).
-   **Comparison Targets:** Identify other sampling algorithms or tools (e.g., MCMC, genetic algorithms, other protein design software) to compare against.
-   **Test Cases:** Create specific protein design problems or fitness landscapes to test the algorithms.

### Evolutionary Studies
For evolutionary studies, you might want to:
-   **Fitness Functions:** Experiment with different fitness functions that represent various evolutionary pressures (e.g., stability, binding affinity, catalytic activity).
-   **Landscape Analysis:** Use the sampled sequences to map and analyze protein fitness landscapes.
-   **Ancestral Reconstruction:** Potentially adapt the SMC framework for ancestral sequence reconstruction or inferring evolutionary histories.

## Development Practices

### Testing
While there's a `tests/` directory, ensure that new features and modifications are accompanied by appropriate unit and integration tests. This is crucial for maintaining code quality and verifying the correctness of complex simulations.

### Linting
The project uses `ruff` for linting. The current configuration focuses on `F` (Pyflakes) and `B` (Bugbear) error codes, and ignores `E501` (line length). Always run `ruff check src/ --fix` before committing changes to ensure code quality and consistency.

### Code Structure
Maintain the modular structure within `src/`. When adding new functionalities, consider where they best fit within the existing `experiment.py`, `initiation.py`, `mpnn.py`, `mutate.py`, `sampling/`, and `utils/` modules, or if a new module is warranted.

### Virtual Environment
Always activate the virtual environment before running any Python commands or scripts. The activation script is located at `.venv/bin/activate`. You should do this before running any Python-related functions.

### Running Tests
To run all tests, use the following command from the project root:
```bash
"/Users/mar/MIT Dropbox/Marielle Russo/2025_workspace/proteinsmc/.venv/bin/python" -m pytest
```
To run a specific test file, provide its path:
```bash
"/Users/mar/MIT Dropbox/Marielle Russo/2025_workspace/proteinsmc/.venv/bin/python" -m pytest "/Users/mar/MIT Dropbox/Marielle Russo/2025_workspace/proteinsmc/tests/utils/test_combined_fitness.py"
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
- Corrected the `translate` function in `src/utils/nucleotide.py` to accurately determine `valid_translation` based on the presence of 'X' residues.
- Fixed the argument order for `mpnn_model.score` in `src/scoring/mpnn.py`.
- Addressed syntax errors and incorrect Glycine codon frequencies in `src/utils/constants.py` to ensure correct CAI score calculation.

**Remaining Work:**
- Resolve the failing `test_calculate_fitness_population_nucleotide_sequences` test in `tests/utils/test_combined_fitness.py`. The current issue is that the calculated CAI scores are still incorrect, even after fixing the constants. This suggests a deeper issue with the CAI calculation logic or the test's expected values.
- Address any remaining test failures in `tests/utils/test_metrics.py` and `tests/utils/test_resampling.py`.

**Challenges Encountered:**
- Persistent issues with `replace` tool due to subtle whitespace differences and multiple occurrences of `old_string`. This required careful manual inspection and precise string matching.
- Debugging JAX-related errors, especially `NameError` and incorrect numerical outputs, required adding `jax.debug.print` statements for intermediate value inspection.
- The `ModuleNotFoundError: No module named 'src'` when running pytest from the virtual environment was resolved by explicitly using the full path to the `python` executable within the virtual environment.
