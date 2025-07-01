# GEMINI.md - Development Guidelines for Protein SMC Experiment

This document provides guidelines and context for continuing development on the Protein SMC Experiment project.

## Project Goals
The primary goals for this repository are:
1.  **Benchmarking:** Evaluate the performance of the JAX-based Sequential Monte Carlo (SMC) sampling approach against other protein sequence sampling methods. This could involve comparing convergence rates, diversity of generated sequences, and computational efficiency.
2.  **Evolutionary Studies:** Utilize the SMC framework to study protein evolution, including exploring fitness landscapes, identifying evolutionary pathways, and understanding the impact of different selective pressures.

## Current Status and Future Work

### Island Models
Based on the current codebase, explicit "island models" for population dynamics (e.g., with migration between islands) are **not** implemented. The existing `sampling/smc.py` contains logic related to "initial population" and "population mutation," but this refers to the population within a single SMC chain.

To incorporate island models for evolutionary studies, you would need to:
-   Implement mechanisms for running multiple independent SMC simulations (islands).
-   Develop a strategy for migration between these islands (e.g., periodically exchanging sequences based on fitness or other criteria).
-   Design a way to manage and track the state of multiple populations.

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
Always activate the virtual environment before running any Python commands or scripts. The activation script is located at `.venv/bin/activate`.

### Shell Commands
When using shell commands, especially when dealing with paths that might contain spaces or special characters, always enclose the paths in quotes.

### Dependencies
Keep `requirements.txt` up-to-date with any new Python package dependencies.

By following these guidelines, we can ensure consistent, high-quality development for the Protein SMC Experiment project.