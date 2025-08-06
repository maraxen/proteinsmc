# Protein SMC WORK IN PROGRESS

Implements a Sequential Monte Carlo (SMC) evolutionary simulation for protein sequence design using JAX, numpy, scipy, pyarrow, and prxteinmpnn. The main experiment logic is modularized under `src/` and can be run as a script.

Multiple samplers have been introduced now, and support for esm and proteinmpnn likelihoods at fitness have been integrated and tested.

Interoperability with translated nucleotide sequences and related scores such as codon adaptation index (currently only E. coli frequencies are included) is also available, perhaps allowing simultaneous selection for both folding and expression.

The fitness functions defined for the samplers can be arbitrarily combined for combinatorial optimization of protein sequences, and Jax makes the underlying computation incredibly performant.

Stay tuned for benchmarks and actual science

## Features

- JAX-accelerated SMC simulation for protein design
- Codon and amino acid mappings, CAI calculation, and MPNN scoring
- **Parallel Replica SMC:** A sophisticated island model implementation for enhanced sampling, allowing for replica exchanges between islands at different temperatures.
- Modular, well-documented code
- Ruff linting enabled (see `pyproject.toml`)

## Setup

1. (Recommended) Create a virtual environment:

   ```zsh
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```zsh
   pip install -r requirements.txt
   ```

## Linting

Run ruff to check and fix code style:

```zsh
ruff check src/ --fix
```

## Project Structure

- `src/` — Main source code
- `requirements.txt` — Python dependencies
- `pyproject.toml` — Ruff and project config
- `.github/copilot-instructions.md` — Copilot custom instructions

## Notes

- Ensure you have the required Python version and JAX/colabdesign dependencies.
- The main experiment entry point is in `src/main.py`.
