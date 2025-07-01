# Protein SMC Experiment (JAX)

This project is a Python codebase converted from a single Jupyter cell. It implements a Sequential Monte Carlo (SMC) simulation for protein sequence design using JAX, numpy, scipy, pyarrow, and colabdesign. The main experiment logic is modularized under `src/` and can be run as a script.

## Features
- JAX-accelerated SMC simulation for protein design
- Codon and amino acid mappings, CAI calculation, and MPNN scoring
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
3. Run the main experiment:
   ```zsh
   python -m src.main
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
