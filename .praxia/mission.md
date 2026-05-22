# proteinsmc — Mission

proteinsmc provides a JAX-accelerated Sequential Monte Carlo (SMC) framework for protein sequence design and evolutionary studies.

**Core goals:**
1. **Benchmarking** — Evaluate SMC, MCMC, HMC, NUTS, and Gibbs samplers against each other on protein fitness landscapes. Metrics: convergence rate, sequence diversity, computational efficiency.
2. **Evolutionary studies** — Map protein fitness landscapes, identify evolutionary pathways, and analyze the impact of different selective pressures using the SMC framework.
3. **Optimal Experiment Design (OED)** — Bayesian optimization of sampler hyperparameters (N, K, q, population size, mutation rate) using GP models and Fisher Information criteria.

**Design principles:**
- **JAX-first:** all hot-path computation uses JIT, vmap, and scan. Sampler state is always a registered PyTree.
- **Polymorphic dispatch:** `runner.py` routes to any sampler via `SAMPLER_REGISTRY` — adding a sampler requires only a config class, a loop function, and a registry entry.
- **Centralized I/O:** all output goes through `io.py`. Runs are identified by UUID; tensors use `.safetensors`, scalars use `.jsonl`.
- **Fitness is composable:** scoring functions are factory-built and vmapped; multiple fitness functions combine via `CombineFunction`.

**What this project is not:**
- Not a protein structure predictor (no folding)
- Not a general deep learning training framework
- Not a sequence database or alignment tool

**Stack:** Python ≥3.11, JAX, Flax, BlackJAX, ProteinMPNN, ESM2, uv, ruff.
