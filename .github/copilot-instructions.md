<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Protein SMC Experiment (JAX) Copilot Instructions

These instructions guide Copilot in understanding the codebase, adhering to best practices, and contributing effectively to the "Protein SMC Experiment (JAX)" project.

## 1. Project Overview and Goals

This project focuses on **Sequential Monte Carlo (SMC) simulations for protein sequence design**, leveraging the **JAX** ecosystem for accelerated computation.

**Primary Goals:**
* **Benchmarking:** Evaluate convergence, diversity, and efficiency of sampling algorithms (SMC, MCMC, HMC, Gibbs) against various metrics and comparison targets.
* **Evolutionary Studies:** Explore protein fitness landscapes, analyze evolutionary pathways, and perform ancestral reconstruction, experimenting with different fitness functions (e.g., stability, binding, catalytic activity).

**Key Features:**
* JAX-accelerated SMC and other MCMC methods.
* Integration of codon/amino acid mappings.
* Implementation of Codon Adaptation Index (CAI) and MPNN scoring.
* Support for Parallel Replica SMC (island model with replica exchange).
* Modular and well-documented code structure.

## 2. Core Principles & Development Practices

Adherence to these principles is paramount for maintaining code quality and consistency:

### A. JAX Idioms and Functional Programming
* **Prioritize JAX-compatible code:** All new code, especially numerical operations, should be written with JAX's functional programming paradigm in mind.
* **Immutability:** Favor immutable data structures (e.g., JAX PyTrees, `equinox.Module`, `flax.linen.Module`) where applicable.
* **JIT/Vmap/Scan Compatibility:** Ensure functions are compatible with JAX's `jit`, `vmap`, and `scan` transformations for performance.
* **Static Arguments:** Utilize `static_argnums` for function arguments that are not JAX types and do not change across JAX transformations (e.g., Python built-in types, strings, tuples of static values, or dataclasses acting as auxiliary data).
* **PyTree Registration and Dataclasses:**
    * Custom data structures used with JAX should be registered as PyTrees.
    * **If a dataclass contains frozen Python built-in types (e.g., `int`, `str`, `tuple`), these dataclasses should be treated as *auxiliary data* (i.e., passed as `static_argnums` to `jax.jit` or other transformations).**
    * **If a dataclass contains JAX types (e.g., `jax.Array`), these dataclasses should *not* be frozen and their JAX type fields should be registered as *children* in the PyTree structure. This allows JAX to trace and transform the internal JAX arrays.**

### B. Code Quality & Linting (Ruff)
* **Linter:** Use **Ruff** for linting.
* **Configuration:** Adhere to the `ruff.toml` settings.
    * `select = ["ALL"]` (all rules enabled by default)
    * `ignore = ["PD"]` (pandas-specific rules are ignored)
    * **`line-length = 100`**: This is the primary target for line length. Ensure generated code adheres to this limit.
    * `indent-width = 2`
    * `fix = true` (Ruff's autofix capabilities should be utilized).
* **Execution:** Run `ruff check src/ --fix` regularly to apply automatic fixes.
* **Fix Failure Threshold:** **If automated `ruff --fix` attempts fail more than 5 times consecutively on the same set of issues, cease further attempts and flag the code for manual review by the user.**

### C. Type Checking (Pyright)
* **Evaluator:** Use **Pyright** for static type checking.
* **Strict Mode:** All code should pass Pyright in **strict mode**. Ensure type hints are accurate and comprehensive to satisfy strict type checking.

### D. Testing
* **Availability:** All new components and features *must* be accompanied by comprehensive **unit and/or integration tests**.
* **Framework:** Use `pytest`.
* **Execution:** Tests are located in the `tests/` directory. Run tests with `python -m pytest tests/`.
* **Test Philosophy:** Tests should cover typical use cases, edge cases, and ensure correctness of JAX transformations where applicable.

### E. Code Structure & Modularity
* **Modularity:** Maintain the modular structure within the `src/proteinsmc/` directory (e.g., `sampling/`, `scoring/`, `utils/`).
* **DRY Principle:** Adhere to the "Don't Repeat Yourself" (DRY) principle. Refactor common logic into reusable functions or modules.
* **Dependencies:** Avoid adding unnecessary external dependencies. Keep `requirements.txt` minimal and up-to-date.
* **Documentation:** Ensure all functions, classes, and complex logic are well-documented with clear docstrings and comments.

## 3. Current Focus & Known Issues

The current primary development focus is on **utilities to improve memory efficiency** within the codebase. Copilot should prioritize this aspect in its code generation and suggestions.

## 4. Copilot Interaction Guidelines

* **Prioritize JAX-idiomatic solutions, including careful consideration of `static_argnums` and dataclass PyTree registration.**
* **Strictly adhere to Ruff linting rules and Pyright strict mode for type checking.**
* **Always include appropriate tests for new code.**
* **If automated fixes for linting or type errors fail repeatedly (more than 5 times for the same issue), report to the user for manual review.**
* Refer to `GEMINI.md` for detailed project context and goals.
* Refer to `README.md` for project setup and general overview.