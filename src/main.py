"""
Main entry point for the Protein SMC Experiment (JAX).
This script runs the main experiment loop as described in the original Jupyter cell.
"""
from .experiment import run_experiment


def main():
    run_experiment()

if __name__ == "__main__":
    main()
