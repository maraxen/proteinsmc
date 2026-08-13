# proteinsmc rebase — final migration plan

> ## ⚠ DO NOT EXECUTE THE MOVE TABLE (§4) AS WRITTEN
>
> A 13-agent lineage audit (260813) checked every cross-repo move, DELETE, and capability-gap claim in this
> plan against real source on both sides. **Of 60 claims, 17 survived unchanged, 14 were refuted, 27 need
> qualification.** See **`260813_proteinsmc-move-table-errata.md`** (Addendum B) before acting on anything below.
>
> The plan's *architecture* survives — the §2 DAG, the xtrax dependency direction, the codon-quarantine
> placement, and "asr owns the OED harness" are all confirmed. The **move table as a set of executable
> instructions** does not.
>
> Two items are actively harmful and block everything else:
> 1. **§4 deletes proteinsmc's alphabet and moves `scoring/mpnn.py` into aminx.** proteinsmc's encoding is
>    **AlphaFold** ordering (`utils/constants.py:10-31`) while three constants are named after ProteinMPNN;
>    `scoring/mpnn.py` performs *no* alphabet conversion yet feeds those ints to an MPNN-ordered scorer.
>    This is already live in asr. Executing §4 relocates a silent-corruption bug into the one repo whose
>    `aa_convert.py` exists to prevent it. See Addendum B §B.2.1 and gate G1.
> 2. **§1's premise that aminx "structurally cannot host" an iterative sampler is false** — the lint block at
>    `aminx/pyproject.toml:143-169` bans four *pipeline* modules, not a *pattern*, and aminx already ships
>    618 LOC of Gibbs + parallel tempering under it. Every downstream row inherits this. See §B.3.17 and A.7.
>
> Also: **five DELETE rows target live code** (§B.4), and `oed/opt.py` — moved to asr as a "real payload" —
> is mathematically broken (rank-1 FIM, `det ≡ 0`, the BO acquisition selects at random). See §B.2.4.

**Status:** superseded in part — see **Addendum B** (lineage audit) and **Addendum C** (independent final review), both in `260813_proteinsmc-move-table-errata.md`. Originally: actionable. Three fatal objections from adversarial review stand and have forced changes to the partition; they are marked **[CONCEDED]** where they appear.

> ⚠ **Read the gate list in Addendum B §B.6 *as revised by Addendum C* before acting on anything below.** The gate order changed: **G0 → G2 → G-α → G1**, not the original G1-first. Addendum C found that B's own lead gate targeted a function that cannot execute (`scoring/mpnn.py` is dead behind a failing `find_spec`) while the live corruption sits at `asr/src/asr/analysis/biophysics.py:66-70`, and that neither document ever decided **what alphabet a proteinsmc integer sequence is** — the question everything else in that group depends on (new gate G-α).
>
> Two items block execution outright: **the alphabet bug** (§4 would relocate a live silent-corruption defect into aminx, the one repo whose `aa_convert.py` exists to prevent it) and **§1's aminx premise below, which is false** — the lint block bans a pipeline, not a pattern, and `aminx/src/aminx/potts/sampling.py` already ships 618 LOC of Gibbs + parallel tempering under it. Every §4 row justified by "aminx cannot host a sampler" inherits that error.
>
> Also: **five DELETE rows target live code**, `oed/opt.py`'s Fisher-information criterion is rank-1 by construction so its `det` is identically zero and the BO acquisition has been selecting at random, and `asr/vendor/proteinsmc` is a *separate checkout* whose only divergent file is the one §4 calls the edge-flip file. B's 17/14/27 claim tallies are a severity summary, not an auditable ledger (Addendum C, C8).
>
> **The architecture in §2 survives. The move table in §4 does not, and must be re-derived row by row — see Addendum D.**

---

## 1. Verdict

**proteinsmc should become the owner of the population-sampling target contract — `api.py` + `registry.py` + `drive.py` — with its kernels as reference implementations, not as the product.** The role "proteinsmc hosts sampling kernels" does not survive inspection: 633 of ~1,700 LOC in `sampling/` is delegation to blackjax (`sampling/particle_systems/smc.py:11-14` imports `blackjax.smc.base.step`; `hmc.py`, `nuts.py`, `mcmc.py` each build a blackjax kernel), and the genuinely native algorithmic content after subtraction is ~1,280 LOC of which the largest single piece — `parallel_replica.migrate` — is **not** the shared replica-exchange primitive the collision matrix claimed. What is real, and what nothing in the ecosystem currently owns, is the contract: a batched, jit-safe `Fitness` object that can be produced by aminx (logits, ESM), by prolix (physics energies), or in-repo (NK), plus a drive loop that iterates a population against it. ~~aminx structurally *cannot* host that — `aminx/pyproject.toml:143-169` enforces ADR `260605_potts-parallel-not-stageset` with flake8-tidy-imports banned-api rules that wall `potts` off from `aminx.inference.decode` / `aminx.host.plan` / `aminx.types.stages`, i.e. aminx has already ruled by lint that a non-StageSet iterative sampler cannot live in its main body.~~ **[REFUTED — Addendum B §B.3.17, gate G11.** That block has sixteen entries, of which four are the potts wall, and it bans a *pipeline*, not a *pattern*. `aminx/src/aminx/potts/sampling.py` already ships 618 LOC of single-site Gibbs and parallel tempering under that same lint, importing none of the banned modules. This is the potts error — a structural signal read as a semantic conclusion — sitting in the plan's own foundation. The conclusion that proteinsmc owns the contract may still hold on other grounds (the L1-leaf dependency argument, the contract-is-the-product argument), but it must be re-argued; G11 blocks §6.1 and the framing of §4's aminx group until it is.** prolix cannot host it either: it consumes `Float[Array,"N 3"]` positions, not int8 sequences. The hole is real; it is just not the hole the brief named.

**The author's intended separation holds at the bottom of the stack and breaks at the top, in three specific assignments.** It holds for xtrax: `prolix -> {proxide>=0.1.0a8, xtrax>=0.4.0a5}` is a proven, working, load-bearing edge at five call sites beyond the adapter shim, and `prolix/tiling/xtrax_adapter.py` is the correct and generalizable resolution pattern for the three-way tiling collision — including for aminx, whose residual fork is a documented, temporary parity baseline (EPIC #1541), not architectural dissent. It breaks in three places. (1) **proxide is not "host processing ops"** — its only Python parallelism is grain prefetch autotuning feeding `create_protein_dataset`, and `proxide-parallel-rt` is a ~25-line AtomicUsize registry; proxide is protein structure I/O + chemistry + a Grain pipeline. Host-parallel ops are already xtrax's, and `aminx.host` / `prolix.api` already treat them that way. (2) **aminx has expanded from "running models" into "running dynamics on models"** — `aminx/ebm/langevin.py` is an MD integrator over backbone coordinates (prolix's domain) and `aminx/potts/sampling.py` implements Gibbs and Hukushima–Nemoto PT. (3) **proteinsmc is not lightweight and has no consumers** — a ~500-line from-scratch ESM-C transformer in `utils/esm.py`, a ~1,500-LOC Bayesian-optimization harness in `oed/`, and `runner.py:303` dispatching only `"smc"` while `SAMPLER_REGISTRY` advertises six samplers. Fix those three assignments and the shape the author drew is right.

---

## 2. Separation of concerns (final)

| Repo | Owns | Explicitly does NOT own | Depends on |
|---|---|---|---|
| **xtrax** | Host↔accelerator boundary: `tiling` (AxisSpec, BatchPlanner, MemoryBudget, Bucket, DedupSpec), `transforms` (safe_map, safe_scan), `stages` (Fuse/Tap/Sink, AxisBoundary, executor), `distributed.sharding`, `run` (SinkSpec, ZarrStagingSink), Trainer/Engine | Research-integrity governance (`xtrax/loop/`'s 26 gate modules → bathos); `xtrax/composition` scaffolding for an unshipped feature; any domain content | jax only |
| **proxide** | Protein structure I/O and parsing, chemistry (`chem/residues.py` = canonical residue ordering), the MPNN↔AF permutation (one backend-parameterized copy), Atom37/AtomicSystem, Grain dataset pipeline, Rust structure ops | Generic host-parallel ops (that is xtrax); any nucleotide/codon content (it has none); sampling | jax, rust crates |
| **bathos** | Experiment provenance: git SHA capture, run catalog, campaigns, attestation, claims, sidecars, cluster submission | Numerics; nothing in a JAX hot path | — |
| **trex** | Phylogenetics: trees, Sankoff, sparse trees, ancestral interactions, ground truth. Vendored asr workspace member (`asr/vendor/trex/src/trex/`) | NK fitness landscapes (concede to proteinsmc — see §8 Q4); batching utilities (`trex.utils.memory.safe_map` → xtrax) | jax |
| **proteinsmc** | **The population-sampling target contract**: `Fitness`, `Proposal`, `SamplerKernel`, `KernelState`, `KernelInfo`, the `FITNESS`/`KERNELS`/`SCHEDULES` registries with entry-point discovery, `drive.sample()`. Reference kernels (SMC, island-migration PRSMC, Gibbs, blackjax MCMC/HMC/NUTS adapters, STE). Beta/temperature schedules. NK landscape. Quarantined codon subpackage | Any model weights, any transformer, any structure; Bayesian optimization / GP; phylogenetics; tiling, bucketing, memory budgeting, sharding; git-SHA capture; an amino-acid alphabet in the core (`q: int` is the whole alphabet story) | jax, equinox, blackjax, jaxtyping, optax, **xtrax**, (flax — see §4 note) |
| **aminx** | Running models: ProteinMPNN/LigandMPNN, **ESM-C** (incoming), Potts/TRW energies, EBM score models, AR decode (`DECODERS` registry, `StageSet`), the sequence→structure bridge when it exists. Publishes `psmc.Fitness` subclasses via the `proteinsmc.fitness` entry-point group | Generic MCMC algorithms (Gibbs/DLMC/PT → proteinsmc, phase 5); coordinate-space integrators (`ebm/langevin.py` → prolix); its own tiling fork (retire after EPIC #1541 flips) | proxide, xtrax, **proteinsmc** |
| **prolix** | Physical scoring and dynamics of structures: force fields, GBSA, integrators (`types/integrators.py`), thermostats, constraints, neighbor lists, XTC trajectory sink, ensemble policy over xtrax. Peer of aminx | Sampling-algorithm ownership (`pt/replica_exchange.py` may delegate, phase 5); tiling mechanism (already correctly adapts) | proxide, xtrax, (proteinsmc — optional, phase 5) |
| **asr** | Ancestral-sequence-reconstruction research harness: OED/Bayesian optimization (`asr/src/asr/oed/`, already forked — `asr/oed/gp.py:3` says "Adapted from proteinsmc/oed/gp.py"), SMC-tree generation, evolutionary trajectory analysis, MTT training pipeline | Being a library; general sampling | proteinsmc, aminx, prolix, trex, proxide, xtrax, bathos |

**No new project is created.** Every candidate has an existing home: OED → asr (which already forked it), `smc_tree.py` → asr (which vendors trex, the package it imports), ESM → aminx (verified capability gap: `grep -rln esm ~/projects/aminx/src/aminx/` returns nothing), codon → stays quarantined in proteinsmc (§5).

### Dependency DAG

```
L0   xtrax        proxide        bathos        trex
L1   proteinsmc -> {xtrax}
L2   aminx      -> {proteinsmc, proxide, xtrax}
     prolix     -> {proxide, xtrax}            [+ proteinsmc, optional, phase 5]
L3   asr        -> {proteinsmc, aminx, prolix, trex, proxide, xtrax, bathos}
```

Topological order `xtrax, proxide, bathos, trex | proteinsmc | aminx, prolix | asr`. Every edge points strictly right → **acyclic**.

- **Against the working edges:** `prolix -> {proxide, xtrax}` is preserved verbatim and untouched by this plan. `aminx -> {proxide, xtrax}` likewise. The only new edge is `aminx -> proteinsmc`, which points from L2 to L1 and cannot close a cycle because `proteinsmc -> aminx` is deleted, not renamed.
- **The cycle risk is `proteinsmc <-> aminx`.** Today `proteinsmc -> aminx` exists solely through `src/proteinsmc/scoring/mpnn.py:22` (`from prxteinmpnn.scoring.score import make_score_sequence`) and `scoring/esm.py`. Enforcement after the flip is structural, not conventional: proteinsmc's `pyproject.toml` must list neither `aminx` nor `prxteinmpnn` **nor an optional extra** (an extra still registers the edge for the resolver and licenses a lazy import), plus a five-line CI grep over `src/proteinsmc` for `aminx|prxteinmpnn|proxide`.
- **`proteinsmc -> proxide` is eliminated,** not relocated. It exists only via `scoring/mpnn.py`'s `from proxide.ops.dataset import create_protein_dataset`, which departs with that file. **I reject importing `proxide.core.types` for the sequence vocabulary** — that re-creates the exact edge being cut to save ~30 lines of jaxtyping aliases, in the one repo we are making model-free. The core stops naming what the integers mean.
- **`aminx <-> prolix`: neither exists, neither should.** Peers: aminx = learned/statistical scoring of sequences; prolix = physical scoring and dynamics of structures.

---

## 3. The sampler contract

**proteinsmc owns the protocol definition.** It lives in one file, `src/proteinsmc/api.py`, replacing both `models/protocols.py` (abandoned, `FitnessFn` takes `dict[str, Array|float|int]`) and the aliases in `models/fitness.py` (`StackedFitnessFn` takes `PyTree|Array`) — the defect being fixed was caused by having two.

### 3.1 The jit boundary is part of the contract [fixes FATAL objection #1]

`Fitness` is **not** a bare `Protocol`. It is a concrete `eqx.Module` base class, and kernels use `eqx.filter_jit`, never `jax.jit(static_argnames=("fitness_fn", ...))`.

This is mandatory, not stylistic. `sampling/particle_systems/smc.py:87-95` and `sampling/particle_systems/parallel_replica.py:254` currently declare `fitness_fn`, `mutation_fn`, `annealing_fn` as **static** argnames, which requires hashability. That works today only because `utils/fitness.py:94` returns a `@jit`-decorated closure hashed by identity. Any scorer carrying weights fails: an `eqx.Module` with a `jax.Array` field raises `TypeError: unhashable type: 'jaxlib._jax.ArrayImpl'`, and `jax.jit(f, static_argnames=('fitness_fn',))` raises `ValueError: Non-hashable static arguments are not supported`. Both scorers that move to aminx already close over arrays — `scoring/esm.py:39-42` closes over a ~500-LOC transformer's weights, `scoring/mpnn.py:104-112` closes over `processed_inputs.coordinates/mask/residue_index/chain_index`. **The central target call site of this whole plan does not compile under the current jit boundary.** Converting `_run_smc_chunk` / `_run_prsmc_chunk` to `eqx.filter_jit` is a phase-2 precondition, before any scorer leaves the repo.

Making `Fitness` a concrete base class rather than a structural Protocol has a second, deliberate effect: aminx must **actually import proteinsmc** to subclass it. A `runtime_checkable` Protocol under `from __future__ import annotations` plus a TOML entry-point group is a dependency edge made of strings; a base class is a real one.

### 3.2 Two fitness forms, not one [fixes SERIOUS objection: batched form is wrong for 3 of 8 kernels]

```python
# proteinsmc/types.py  (~30 lines, no proxide import)
Population = Int[Array, "n L"]                 # int8, values in [0, q)
Relaxed    = Float[Array, "n L q"]             # for STE / HMC / NUTS
Scores     = Float[Array, "n"]
LogWeights = Float[Array, "n"]
Ctx        = PyTree      # {"beta": f32[], "step": i32[], "axis_size": i32[]}
```

```python
# proteinsmc/api.py
class Fitness(eqx.Module):
  """Batched target. Array leaves are TRACED; config leaves use eqx.field(static=True).
  Consumed under eqx.filter_jit — never as a jax.jit static argument."""
  def __call__(self, key: PRNGKeyArray, x: Population | Relaxed, ctx: Ctx) -> Scores: ...

class PointFitness(eqx.Module):
  """Single-sequence target. For kernels whose batch axis is not n_particles."""
  def __call__(self, key: PRNGKeyArray, x: Int[Array, "L"], ctx: Ctx) -> Float[Array, ""]: ...

def lift(f: PointFitness, *, batch_size: int | None = None) -> Fitness: ...   # xtrax.safe_map
def pointwise(f: Fitness) -> PointFitness: ...                                # size-1 leading axis
```

Both directions are required. `sampling/gibbs.py:64-80` builds `proposals = jnp.tile(seq, (n_states,1)).at[:,pos].set(jnp.arange(n_states))` and vmaps over `n_states` variants of **one** sequence — a scorer planning tiling "against 512 particles" would be planning against the wrong axis. `sampling/hmc.py:57-60` needs a **scalar** log-density plus its gradient for `blackjax.hmc.build_kernel()`. So: `Ctx` carries `axis_size`, each kernel declares which form it consumes, and the tiling argument is restated correctly as *"the scorer owns tiling over whatever axis the kernel presents."*

The batched form remains the default because the justification is sound where it applies: an MPNN scorer over 512 particles must choose Vmap / SafeMap / Bucket itself via `xtrax.tiling.BatchPlanner` against a real memory budget, and a caller-side `jax.vmap` makes demotion to a chunked map structurally impossible. This also resolves a live inconsistency — `CLAUDE.md` claims scorers take a batch, `models/fitness.py:15` types them per-sequence, both half-true.

### 3.3 Kernel, state, proposal

```python
class KernelState(eqx.Module):
  x: Population | Relaxed
  log_w: LogWeights
  score: Scores
  step: Int[Array, ""]
  inner: PyTree = None      # kernel-private: blackjax state, momenta, island axis,
                            # per-particle tuned `update_parameters`

class KernelInfo(eqx.Module):
  log_z_inc: Float[Array,""]; ess: Float[Array,""]; accept_rate: Float[Array,""]
  beta: Float[Array,""];      mean_score: Float[Array,""]
  extra: dict[str, Array]     # per-step DIAGNOSTICS ONLY. Nothing here is read back
                              # into the loop; anything a kernel consumes lives in `inner`.

class SamplerKernel(eqx.Module):
  def init(self, key, x0, fitness: Fitness, ctx: Ctx) -> KernelState: ...
  def step(self, key, state: KernelState, fitness: Fitness, ctx: Ctx
           ) -> tuple[KernelState, KernelInfo]: ...
  def skeleton(self, shapes: ShapeSpec) -> KernelInfo: ...   # for deserialisation, see §3.5

class Proposal(eqx.Module):
  """Kernel-PRIVATE. Returns (population, aux) — the tuple shape blackjax's
  smc.base.step update_fn requires; per-particle tuned params ride in `params`."""
  def __call__(self, key, x: Population, params: PyTree, ctx: Ctx
               ) -> tuple[Population, PyTree]: ...
```

`Proposal` carries `params` and returns a tuple because `runner.py:149-171`'s `mutation_fn(key, particles, update_parameters: dict) -> tuple[Array, dict]` reads `update_parameters["mutation_rate"]`, and `smc.py:104` wires it as blackjax's `update_fn`. A bare `(key, x, ctx) -> Population` protocol cannot be adapted to `blackjax.smc.base.step` without inventing a slot for the tuned-parameter channel; `kernels/_blackjax.py` owns the lift.

`step` takes **no IO parameter**. This deletes four hand-threaded `io_callback: Callable | None` params (`sampling/{gibbs,mcmc,hmc,nuts}.py`) plus a fifth in `io.py`. Streaming becomes `sample(..., boundary=xtrax.stages.Tap(sink))`, the only implementation that documents the two failure modes proteinsmc's call sites have no defense against ("Cannot vmap ordered IO callback"; the silent SafeMap `batch_size` override under `ordered=True`).

```python
# proteinsmc/drive.py
def sample(kernel: SamplerKernel, fitness: Fitness, x0, steps: int, key,
           ctx_fn: Callable[[Int[Array,""]], Ctx] | None = None,
           boundary: xtrax.stages.AxisBoundary | None = None
           ) -> tuple[KernelState, KernelInfo]: ...
```

### 3.4 The four fitness sources under one protocol

```python
# in aminx — aminx/fitness/mpnn.py
class MPNNFitness(psmc.Fitness):
  model: eqx.Module                                   # traced weights
  struct: proxide.AtomicSystem                        # traced arrays
  decoding: str = eqx.field(static=True, default="random")
  def __call__(self, key, x, ctx):
    return aminx.scoring.score.score_batch(key, self.model, self.struct, x, self.decoding)

# in aminx — aminx/fitness/esm.py   (after utils/esm.py lands there)
class ESMFitness(psmc.Fitness):
  esmc: aminx.models.esmc.ESMC
  def __call__(self, key, x, ctx):
    return aminx.models.esmc.pseudo_loglik(self.esmc, aminx.utils.aa_convert.mpnn_to_esm(x))

# in proteinsmc — landscapes/nk.py  (zero external deps: the reference implementor)
class NKFitness(psmc.Fitness):
  landscape: NKLandscape
  def __call__(self, key, x, ctx): return calculate_nk_fitness_population(self.landscape, x)
```

**The prolix case — the one that crosses the sampling/physics boundary:**

```python
# in prolix — prolix/fitness.py
class PhysicsFitness(psmc.Fitness):
  params: prolix.types.SystemParams          # traced: force-field arrays
  fold: Callable[[Population], Float[Array, "n A 3"]] = eqx.field(static=True)
  relax_steps: int = eqx.field(static=True, default=0)

  def __call__(self, key, x, ctx):
    coords = self.fold(x)                                        # INJECTED, not imported
    if self.relax_steps:
      coords = prolix.physics.simulate.minimise(coords, self.params, self.relax_steps)
    e = jax.vmap(prolix.physics.potential_energy, in_axes=(0, None))(coords, self.params)
    return -e                                                    # energy → fitness
```

`fold` is a **caller-injected callable**, never an import. That is what keeps `prolix -> aminx` from appearing in the DAG while still letting the user write `PhysicsFitness(params, fold=aminx.ebm.structure_prediction.fold_batch)`. **This case does not work today** and the plan does not pretend otherwise: nothing in the ecosystem turns an int8 sequence into coordinates (§8 Q6). The contract is written so that when aminx builds that bridge, no repo boundary has to move.

Registration is entry-point based, so aminx adds scorers without proteinsmc knowing it exists:

```toml
# aminx/pyproject.toml
[project.entry-points."proteinsmc.fitness"]
mpnn = "aminx.fitness.mpnn:MPNNFitness"
esm  = "aminx.fitness.esm:ESMFitness"
```

### 3.5 Serialisation [fixes SERIOUS objection: `extra` breaks the reader]

`io.py:162,197` calls `eqx.tree_deserialise_leaves(buffer, skeleton)`, and the skeleton is built by `utils/serialization.py::create_sampler_output_skeleton(config: BaseSamplerConfig)` — both of which this plan deletes. `eqx.tree_serialise_leaves` writes bare leaves with no structure, so "records are self-describing" is false for the serialiser in use. Two changes, both in phase 2:

1. `SamplerKernel.skeleton(shapes) -> KernelInfo` — the kernel is the only thing that knows its own `extra` keys.
2. `io.py` writes a small JSON header (kernel name, `q`, `L`, `n`, leaf shapes) alongside each ArrayRecord. `msgpack-numpy>=0.4.8` is **already** a declared dependency (`pyproject.toml:24`); switching the payload encoding to msgpack is the cheaper alternative and removes the skeleton requirement entirely. **Existing `.arrayrecord` outputs become unreadable** — `io.py` has no versioning. Say so in the changelog.

### 3.6 Target call site

```python
import jax, aminx, proteinsmc as psmc

fit = psmc.weighted(
    (aminx.fitness.MPNNFitness(model, struct),        1.0),
    (aminx.fitness.ESMFitness(esmc_300m),             0.3),
    (psmc.landscapes.NKFitness(nk),                   0.1),
)

out, info = psmc.sample(
    kernel  = psmc.kernels.SMC(n_particles=512, resampling="systematic",
                               proposal=psmc.proposals.PointMutation(rate=0.01, q=20)),
    fitness = fit,
    x0      = psmc.proposals.tile_init(seed, n=512),
    steps   = 200,
    key     = jax.random.key(0),
    ctx_fn  = psmc.schedules.linear(0.0, 1.0, steps=200),
    boundary= xtrax.stages.Tap(psmc.io.array_record_sink("outputs/run.arrayrecord")),
)
```

No config object, no registry string, no `sequence_type`, no mesh, no output directory. `import proteinsmc` pulls in jax + xtrax and nothing else.

---

## 4. File-level move table

All 66 modules in `src/proteinsmc/`. Destinations: **STAY**, `repo/path`, **DELETE**, or **REWRITE**.

### Top level (5)

| File | Dest | Reason |
|---|---|---|
| `__init__.py` | REWRITE | Curated surface: `Fitness`, `Proposal`, `SamplerKernel`, `sample`, `kernels`, `compose`, `schedules`, `proposals`, `landscapes` |
| `io.py` | STAY, minus git-SHA | `create_metadata_file`'s git shell-out → bathos (1 of 5 such sites in the ecosystem). Payload moves to msgpack (§3.5). Later: register as an `xtrax.SinkSpec` backend, after checking `proxide/io/streaming/array_record.py` |
| `padding.py` | DELETE | Verified dead (zero importers). `SEQ_BUCKETS=(200,400,800,1200)` is a 5th bucket-ladder copy; `xtrax.tiling.bucket` owns `Bucket`/`select_bucket`/`bucketize` |
| `runner.py` | STAY as deprecated shim | Fix the `:303` single-branch dispatch in phase 0, then reduce to `config -> kernel -> drive.sample()`. Delete after one release. asr never calls it |
| `types.py` | STAY, trimmed to ~30L | Keep `Population/Relaxed/Scores/LogWeights/Ctx/PRNGKey`. Drop `ProteinSequence`/`NucleotideSequence`. Deliberately **not** importing `proxide.core.types` |

### `models/` (17) — dissolved into `api.py` + `kernels/` + `schedules.py`, with a deprecated `proteinsmc.models` re-export shim for one release

| File | Dest | Reason |
|---|---|---|
| `models/__init__.py` | REWRITE → shim | Re-exports new locations, emits `DeprecationWarning`. 45 of ~50 test files import from here |
| `models/protocols.py` | DELETE → `api.py` | Abandoned second contract; signatures contradict `models/fitness.py`; referenced only as a TYPE_CHECKING import |
| `models/fitness.py` | SPLIT → `api.py` | Aliases become the one contract. `FitnessFunction`/`CombineFunction`/`FitnessEvaluator` string-keyed indirection deleted. `needs_translation(n_states)` — the mechanism by which codons leak into the fitness core — dies here |
| `models/sampler_base.py` | SPLIT | `SamplerState` → `api.KernelState`; `SamplerOutput` (24-field union, HMC fields on SMC runs) → `api.KernelInfo`. `BaseSamplerConfig` DELETED: `jax.sharding.Mesh` → xtrax, `MemoryConfig` → xtrax, `sequence_type` → `codon/`, `fitness_evaluator` → a value at the call site |
| `models/smc.py` | → `kernels/smc.py` | Config merges into the kernel object |
| `models/parallel_replica.py` | → `kernels/island_migration.py` | Renamed to what it is (§6) |
| `models/gibbs.py` | → `kernels/gibbs.py` | " |
| `models/mcmc.py` | → `kernels/mcmc.py` | " |
| `models/hmc.py` | → `kernels/hmc.py` | Gains required `relax:` boundary |
| `models/nuts.py` | → `kernels/nuts.py` | " |
| `models/annealing.py` | → `schedules.py` | 47L dataclass over a 264L registry; the split is ceremony |
| `models/memory.py` | DELETE | → `xtrax.tiling.MemoryBudget` + a proteinsmc byte-estimator closure (the prolix/aminx precedent) |
| `models/mutation.py` | → `api.Proposal` | 11-line Callable alias becomes a real protocol |
| `models/nk_landscape.py` | → `landscapes/nk.py` | Merged with `utils/nk_landscape.py` + `scoring/nk.py`. **Port `PyTreeNode` → `eqx.Module`** (see flax note) |
| `models/translation.py` | → `codon/translation.py` | `TranslateFuncSignature` → `codon.through_translation(fitness)` — same seam, on the codon side |
| `models/types.py` | DELETE | Self-described "legacy redirect" to `types.py` — an internal second redirect layer |
| `models/lineage.py` | DELETE | One-line docstring stub |

> The recon brief's "7 duplicated pairs" framing is wrong: `models/X.py` is config/type declaration (~250L total), `utils/X.py` is implementation (~1000L), nothing on either side is dead. The dissolution above is colocation-by-concern once a kernel *is* its config — not de-duplication. It is the most optional item in this plan (§8 Q2).

### `sampling/` (10) → `kernels/`

| File | Dest | Reason |
|---|---|---|
| `sampling/__init__.py` | REWRITE → `kernels/__init__.py` | |
| `sampling/particle_systems/__init__.py` | DELETE | Flattened |
| `sampling/particle_systems/smc.py` | → `kernels/smc.py` | Pure JAX + blackjax over opaque callables. Hand-rolled Python chunking loop and `writer_callback` replaced by `SMCKernel.step()` under `drive.sample()`. **`jax.jit(static_argnames=...)` → `eqx.filter_jit`** |
| `sampling/particle_systems/parallel_replica.py` | → `kernels/island_migration.py` | **Not split into a shared exchange kernel** (§6). Fix the `lax.cond` non-scalar-predicate bug in `migrate` against the real stacked fitness; `eqx.filter_jit` |
| `sampling/gibbs.py` | → `kernels/gibbs.py` | Drops private `io_callback`; consumes `PointFitness`; `sequence_length` moves to the kernel constructor (today `make_gibbs_update_fns` materialises L closures at construction time) |
| `sampling/mcmc.py` | → `kernels/mcmc.py` | Drops io_callback; adopts `_blackjax.make_blackjax_log_prob_fn` instead of its inline wrapper |
| `sampling/hmc.py` | → `kernels/hmc.py` | Drops io_callback; **requires explicit `relax: Callable[[Population], Relaxed]`** rather than jax.grad-ing an int8 array. Resolves the 4 known `test_initialization_factory.py` failures by construction |
| `sampling/nuts.py` | → `kernels/nuts.py` | Same relaxation boundary. Docstring self-describes as "a conceptual placeholder"; keep **unregistered** until it has a parity test |
| `sampling/ste.py` | **REWRITE** → `kernels/ste.py`, unregistered | Not a move. `ste.py:18` uses `jax.example_libraries.optimizers` (demo-only upstream API); must be rebuilt on optax. Also takes no PRNG key, no population axis, no fitness, no state. Register only after rewrite + parity test |
| `sampling/initialization_factory.py` | DELETE | A single dispatch on `sampler_type` string — exactly the switch the KERNELS registry removes. Absorbed into each `Kernel.init()`; `sequence_type` threading → `codon/` |

### `scoring/` (6) — the edge flip

| File | Dest | Reason |
|---|---|---|
| `scoring/__init__.py` | DELETE | |
| `scoring/nk.py` | → `landscapes/nk.py` | Merged (3 files → 1) |
| `scoring/combine.py` | → `compose.py` | `make_sum_combine`/`make_weighted_combine` → first-class `sum_of`/`weighted`, not registry-string-selected |
| `scoring/mpnn.py` | → **aminx** `aminx/fitness/mpnn.py` | **THE edge-flip file.** Imports `prxteinmpnn.scoring.score`, `prxteinmpnn.utils.decoding_order` AND `proxide.ops.dataset` — single-handedly responsible for both sibling dependencies. Logic is already a correct thin adapter, just on the wrong side. In aminx: local imports, the `find_spec("prxteinmpnn")` gate **deleted not renamed**, becomes an `MPNNFitness(psmc.Fitness)` + entry point |
| `scoring/esm.py` | → **aminx** `aminx/fitness/esm.py` | Follows its model |
| `scoring/cai.py` | → `codon/cai.py` | Hardcoded against `ECOLI_CODON_FREQ_JAX`; meaningless without codons |

### `oed/` (10) → asr

| File | Dest | Reason |
|---|---|---|
| `oed/__init__.py` | DELETE | |
| `oed/gp.py` | → **asr** `asr/oed/gp.py` | asr's fork (ARD kernel, masked static shapes) is more advanced — reconcile, don't overwrite |
| `oed/structs.py` | → **asr** `asr/oed/structs.py` | asr has `ASRDesign` |
| `oed/opt.py` | → **asr** `asr/oed/opt.py` | Finite-difference FIM determinant; real payload, asr lacks it |
| `oed/phase.py` | → **asr** `asr/oed/phase.py` | Real payload |
| `oed/run.py` | → **asr** `asr/oed/run.py` | The BO outer-loop CLI |
| `oed/nk.py` | → **asr** `asr/oed/nk.py` | Must import `proteinsmc.landscapes.NKLandscape`, not redefine it (pytree identity, see flax note) |
| `oed/experiment.py` | → **asr** `asr/oed/experiment.py` | asr already has this file; merge. `convert_design_to_config` always builds an NK evaluator — the harness has no protein content |
| `oed/tracking.py` | DELETE → bathos | A private JSONL manifest + checkpoint system explicitly "independent of io.py" — proteinsmc carries two run-tracking systems and bathos is a third the author's own rules mandate |
| `oed/smc_tree.py` | → **asr** `asr/oed/smc_tree.py` | **Unimportable today** (`from trex import nk_model`, `trex.types.Adjacency`, `trex.utils.memory.safe_map`; trex is in no proteinsmc pyproject). It is asr code in the wrong repo. Moving it makes the import legal for the first time. Its residual `from proteinsmc.utils import mutation` / `from proteinsmc.utils.jax_utils import chunked_map` (`smc_tree.py:13-14`) must be repointed to `psmc.proposals.PointMutation` and `xtrax.transforms.safe_map` **in phase 2**, before the move — see §7 |

### `utils/` (18) — dissolved, with a deprecated `proteinsmc.utils` shim

| File | Dest | Reason |
|---|---|---|
| `utils/__init__.py` | REWRITE → shim | |
| `utils/esm.py` | → **aminx** `aminx/models/esmc.py` | 498-line from-scratch Equinox ESM-C (Linear, LayerNorm, RotaryEmbedding, MHA, TransformerStack, HF download, custom (de)serialisation) living in `utils/` in the sampling repo. Simultaneously the clearest misplaced-ownership violation and a verified aminx capability gap. proteinsmc keeps **nothing**, not even an adapter |
| `utils/fitness.py` | SPLIT → `compose.py` + `registry.py` | The `lax.switch`/vmap composition survives. `FITNESS_FUNCTIONS` — a module-level dict with four literal `from proteinsmc.scoring import cai, combine, esm, mpnn, nk` imports — becomes a decorator registry + `importlib.metadata.entry_points("proteinsmc.fitness")`. **Those four imports are what makes proteinsmc depend on aminx; deleting them is what breaks the edge.** The `[combined, s1, s2, ...]` concatenated return (`fitness.py:114`) is replaced by `Scores` + per-component scores in `KernelInfo.extra` |
| `utils/annealing.py` | → `schedules.py` | `ANNEALING_REGISTRY` + `@register_schedule` is the ecosystem's only schedule mechanism with an extension point. Absorbs prolix's geometric/linear ladders |
| `utils/nk_landscape.py` | → `landscapes/nk.py` | Alphabet-agnostic q-state kernel; the in-tree, zero-dependency reference implementor of `Fitness` |
| `utils/mutation.py` | SPLIT | `mutate()` (offset-mod-q) → `proposals.PointMutation(rate, q)`; `diversify_initial_protein_sequences` → `proposals.uniform_init` generalised to q; `diversify_initial_nucleotide_sequences` + `_revert_x_codons_if_mutated` → `codon/proposals.py`; `chunked_mutation_step` chunking → `xtrax.transforms.safe_map` |
| `utils/constants.py` | SPLIT / DELETE | Codon section (~180L) → `codon/tables.py`. ESM section (`ESM_AA_CHAR_TO_INT_MAP`, `ESM_BOS_ID`, `ESM_EOS_ID`, `PROTEINMPNN_TO_ESM_AA_MAP_JAX`) → **aminx**, next to `aminx/utils/aa_convert.py`. Protein alphabet (`AA_CHAR_TO_INT_MAP`, `restypes`, `PROTEINMPNN_X_INT`) → DELETE; the core needs only `q: int`, canonical ordering already lives at `proxide/chem/residues.py` |
| `utils/metrics.py` | SPLIT | SMC-intrinsic half (`safe_weighted_mean`, `calculate_logZ_increment`, ESS) → `kernels/diagnostics.py`. Analysis half (shannon/position entropy, KL/Jeffreys/JS, barrier-crossing, geometric mean) → **asr**, its only consumer |
| `utils/memory.py` | DELETE | `suggest_batch_size_heuristic` + `benchmark_batch_size` + `auto_tune_batch_size` pick batch size by wall-clock benchmarking at runtime. xtrax reads the real numbers: `device_memory_budget` (allocator `bytes_limit`), `lowered_memory_estimate` (`Compiled.memory_analysis()`) |
| `utils/jax_utils.py` | SPLIT/DELETE | `chunked_map` → `xtrax.transforms.safe_map` (**caution:** xtrax raises `ValueError` when `n % batch_size != 0`; `lax.map(batch_size=)` does not — verify against `smc_tree`'s `inference_batch_size=64` default before swapping). `generate_jax_uuid`/`generate_jax_hash` → DELETE; run identity is bathos's job |
| `utils/jax_uuid.py` | DELETE | Byte-for-byte copy of both functions in `jax_utils.py`; not exported, imported nowhere |
| `utils/pmap_utils.py` | DELETE | → `xtrax.distributed.sharding`. Manual reshape + pmap + unshard, while `BaseSamplerConfig` built a real `jax.sharding.Mesh` — two incompatible schemes in one repo. Called by no other src file |
| `utils/translation.py` | → `codon/translation.py` | `nucleotide_to_aa`, `aa_to_nucleotide`, `string_to_int_sequence` |
| `utils/initiate.py` | SPLIT | `generate_template_population` tiling → `proposals.tile_init`; `sequence_type` conversion → `codon/init.py` |
| `utils/serialization.py` | DELETE | `create_sampler_output_skeleton` exists only because `SamplerOutput`'s 24 optional fields make deserialisation need a config-derived skeleton. Replaced by `SamplerKernel.skeleton()` + msgpack (§3.5) |
| `utils/blackjax_utils.py` | → `kernels/_blackjax.py`, **ADOPT** | The one orphan worth keeping: mcmc/hmc/nuts each hand-roll the same logdensity wrapper — a 4th intra-repo duplication. It is exactly the `Fitness → blackjax (log_density, aux)` adapter the contract needs, and the home for the `Proposal → update_fn` lift |
| `utils/key_management.py` | DELETE | `split_key_for_sampler`/`split_key_batched`: tested, imported by nothing. **Rejecting "adopt it"** — adoption changes the PRNG stream and invalidates every golden-value test for zero functional gain |
| `utils/config_unpacker.py` | DELETE | Signature-introspection auto-forwarding from a `BaseSamplerConfig` that is itself deleted. Tested, unused, hostile to a legible contract |

**flax note (fixes SERIOUS objection).** `models/nk_landscape.py:8,33` uses `flax.struct.PyTreeNode`; `utils/nk_landscape.py:17,34` uses `flax.struct.dataclass`. Both feed `landscapes/nk.py`, which stays. Either keep `flax` in the dependency list, or — recommended — **port `NKLandscape` and `InteractionCarry` to `eqx.Module` in phase 1**, cheap and before anything imports them cross-repo. Flax `PyTreeNode` registration is keyed on class identity, so it is a phase-3 acceptance criterion that `asr/oed/nk.py` **imports** `proteinsmc.landscapes.NKLandscape` rather than redefining it, or the two become distinct registered pytree types that will not unify across a `scan`/`cond` boundary.

---

## 5. The nucleotide question

**Recommendation: keep it, as `proteinsmc/codon/`, quarantined by a ruff `flake8-tidy-imports` banned-api rule forbidding any import of `proteinsmc.codon` from `api.py`, `drive.py`, `kernels/`, `compose.py`, or `registry.py`.**

The 13-file nucleotide path (`utils/translation.py`, `utils/constants.py` codon section, `scoring/cai.py`, `utils/mutation.py`'s nucleotide branch, `models/translation.py`, plus the `sequence_type` threading through `models/sampler_base.py`, `sampling/initialization_factory.py`, `runner.py`, `utils/initiate.py`, `oed/experiment.py`, `types.py`) collides with **nothing** in the ecosystem — proxide, aminx and prolix have zero codon/nucleotide/DNA/RNA handling; every type alias, field name and padding routine in all three is residue- or atom-indexed. So extraction has no destination and must create one.

Three things make the subpackage the right call:

1. **The separability the extraction would buy already exists in the code.** Only 5 files are unambiguously nucleotide-only; 3 are protein-only; the rest pass `sequence_type`/`n_states` through as opaque parameters and need zero changes. The coupling runs through one callable — `TranslateFuncSignature`, threaded by `utils/fitness.py` — which becomes `codon.through_translation(fitness) -> Fitness`. A repo boundary adds nothing the callable seam does not already provide.
2. **A banned-api rule buys the actual guarantee.** What we want is a type-level proof that the core is alphabet-agnostic — that `q: int` is the whole alphabet story. A lint rule delivers exactly that, at zero packaging cost, using the same mechanism aminx already runs for its potts/StageSet wall (`aminx/pyproject.toml:143-169`, ADR `260605`). Deleting `sequence_type: Literal["protein","nucleotide"]` from every core signature is the other half; the alphabet is a property of the fitness function, not of the sampler.
3. **Maintenance capacity is the binding constraint.** `~/projects/` holds ~90 entries including `demistify-worktrees`, `proxide-worktrees`, `praxia-wt`, `.stale_backup_`, `_pre_filterrepo_backup_`; prolix's repo root is dozens of loose `debug_*.py` / `sprint_b_*.sh`; asr's root is bare-hash directories. Six repos already outrun capacity. A seventh with one consumer is not affordable.

**The counter-argument, and why it loses.** A `codonx` repo would make the guarantee structural rather than lint-enforced, and would give nucleic-acid work a place to grow — the CAI table, codon-usage tables for organisms beyond E. coli, wobble-aware proposals, splice constraints. That is a real future. It loses **today** on two counts: the guarantee is already obtainable for the price of six lines of TOML, and the growth argument is speculative while the maintenance evidence is not. It wins the moment the author commits to serious nucleic-acid work — at which point `proteinsmc/codon/` lifts out cleanly, because the banned-api rule will have kept the seam honest in the meantime. **That is the point of choosing the subpackage: it is the cheap option that does not foreclose the expensive one.**

---

## 6. Resolved collisions

### 6.1 aminx already hosts samplers — resolved by naming the abstractions apart, then composing them

`aminx.registry.SAMPLERS` is a real `Registry[SamplerFactoryFn]` with an `@SAMPLERS.register(key)` decorator, resolved by `SamplingDriver.build_sampler_fn` — but `SamplerFn` is **decode-shaped**: `(prng_key, bundle: InferenceBundle, config, stage_set) -> (ProteinSequence, Logits, DecodingOrder)`. One sequence per call from an AR pass. No population, no weights, no resampling, no generations. `proteinsmc.SAMPLER_REGISTRY` is population-shaped. These are genuinely different abstractions and merging them is wrong.

Resolution, in three parts:

1. **Rename `aminx.registry.SAMPLERS -> DECODERS`** (keep `SAMPLERS` as a deprecated alias). One aminx-internal rename; makes the boundary self-enforcing at the name level.
2. **proteinsmc's `KERNELS` registry becomes real** — decorator, exported protocol, entry-point discovery — and `runner.py:303` is fixed so the six advertised samplers actually run (phase 0).
3. **`decoder_proposal(decoder, bundle, cfg, stages) -> psmc.Proposal`**, shipped in aminx with an ADR in both repos. *An aminx decoder is a proposal distribution; proteinsmc is what turns a proposal into a sampler with respect to an arbitrary target.* Neither registry loses; they compose.

Ship an ADR in proteinsmc recording the absorb-into-aminx counterproposal and its rebuttal, citing `aminx/pyproject.toml:143-169` — aminx has already ruled by enforced lint that a non-StageSet iterative sampler cannot live in its main body, so absorbing proteinsmc's population samplers would either join that quarantine (a repo inside a repo) or break ADR `260605`. Close the question with evidence so it stops reopening.

### 6.2 Three-way tiling (xtrax vs aminx vs prolix) — yes, `xtrax_adapter.py` is the answer for aminx too

**xtrax owns the mechanism, unambiguously.** It is the stated owner of the host↔accelerator boundary and the only implementation with compiler-derived memory budgeting (`device_memory_budget` reads the allocator's `bytes_limit`; `lowered_memory_estimate` reads `Compiled.memory_analysis()`).

**`prolix/tiling/xtrax_adapter.py` generalizes.** ~180 lines: a bidirectional type-mapping pair (`prolix_axis_to_xtrax` / `xtrax_decision_to_prolix`) plus one delegating entry point (`plan_axes_with_xtrax`), zero duplicated strategy-selection logic, memory estimator supplied as a caller closure.

**The brief's characterization of aminx as the unresolved-fork half is a stale snapshot.** `aminx/tiling/__init__.py`'s own docstring says planning/carry/dedup moved to `xtrax.tiling` under EPIC #1541; `planner.py`, `axes.py`, `eda.py`, `bucketing.py` import `AxisSpec`/`BatchPlanner`/`MemoryBudget` directly from xtrax; and `tiling/dispatch.py` already contains `_strategy_to_xtrax` — aminx has independently written the same bidirectional mapping prolix wrote. What remains is ~600 LOC of *deliberately retained* parity baseline: `iterator.py`'s docstring says "Zero production consumers... Kept because tiling/dispatch.py's make_axis_dispatch (also production-dead, also deliberately kept) still constructs these"; `dispatch.py`'s says the native path is the Measurement Protocol baseline `make_axis_dispatch_via_xtrax` is benchmarked against, "Not yet wired into any call site." That is a migration in flight with a justified temporary baseline. **aminx retires the fork when the flip lands; the adapter is the answer for all three consumers.**

**proteinsmc has no such reason.** It reimplements the concern three more times — `utils/jax_utils.chunked_map` (a bare `lax.map` wrapper, a strict subset of `xtrax.transforms.safe_map`), `utils/pmap_utils.distribute` (manual reshape + pmap, a cruder `xtrax.distributed.sharding`), `padding.py` (`SEQ_BUCKETS`, dead) — and was about to add a fourth via `trex.utils.memory.safe_map`. All three delete; a ~150-line `_xtrax.py` adapter modeled on prolix's replaces them (phase 4).

### 6.3 `prolix.pt` replica exchange vs `proteinsmc.parallel_replica` — **[CONCEDED: these are different algorithms]**

The collision matrix asserts "the three differ only in what state they carry; the swap math is identical in all three." **That is false, and the state-agnostic `exchange(key, energies[R], betas[R], phase) -> (perm, accept_edges)` kernel — billed as the plan's key extraction — is withdrawn.**

Verified in `src/proteinsmc/sampling/particle_systems/parallel_replica.py:84-155`, `migrate()` is an **island-model migration operator**, not Hukushima–Nemoto replica exchange:

- Source island is chosen **fitness-proportionally**: `probs_idx1 = jax.nn.softmax(jnp.nan_to_num(mean_fitness, nan=-jnp.inf))`, `idx1 = jax.random.choice(..., p=probs_idx1)`; destination is `idx2 = (idx1 + randint(1, n_islands)) % n_islands` — **any** island, not a neighbor. There is no parity/phase concept anywhere in the function.
- It swaps two **individual particles**, not replicas: `particle_idx = jax.random.randint(key, (2,), 0, population_size)`, then a two-index scatter. The result is not a permutation and cannot be applied by `jax.tree.map(lambda a: a[perm], state)`.
- It is a sequential `fori_loop` over `n_exchange_attempts`, each attempt seeing the previous swap's result — **no single permutation exists**.
- It **calls** `fitness_fn` inside the loop (`fitness1 = fitness_fn(key_acceptance, jnp.expand_dims(particle1,0), all_betas[idx1])`), so it cannot be fed precomputed `energies`.
- Its acceptance carries a `meta_beta` tempering-of-the-tempering factor neither sibling has.

The siblings do not fit the proposed signature either: `aminx/potts/sampling.py:289-320` `_parallel_tempering_exchange` runs both parities in one call via a `nonlocal`-mutating closure over a static `k_rep`, so `phase` is meaningless to it; `prolix/pt/replica_exchange.py:104-115` `attempt_exchange(state, temperatures, key, energy_fn)` picks parity randomly *inside* and takes `energy_fn`, not energies. Shipping the proposed signature would add a **fourth** implementation while unifying zero.

**Resolution:**

- proteinsmc keeps its algorithm as `kernels/island_migration.py`, named for what it is. Its documented single-island edge case and the `lax.cond` non-scalar-predicate bug (§7 phase 0) are fixed there.
- A neighbor-parity `kernels/replica_exchange.py` is written **only after** a three-way parity harness: feed `aminx.potts._parallel_tempering_exchange`, `prolix.pt.attempt_exchange`, and a candidate proteinsmc implementation the same betas/energies on a synthetic ladder and show agreement. If they agree, the shared kernel ships and both siblings delegate. If they disagree, the honest output is *"proteinsmc owns island-model migration; aminx and prolix own neighbor-parity PT; these are different algorithms"* — a real finding.
- The shared kernel, if it ships, returns `(perm, accepted)` **plus** takes a caller-supplied `apply: Callable[[Perm, S], S]`, so no caller blanket-tree_maps. Blanket tree_map would index scalar leaves: `SamplerState.step` is shape `()`, and blackjax NamedTuple leaves are not all replica-major.
- **The collision matrix entry is wrong as written and should be corrected in the record.**

Temperature/beta ladders **do** unify cleanly and are unaffected: `prolix/pt/temperature.py::generate_temperature_ladder` (geometric/linear) registers into proteinsmc's `ANNEALING_REGISTRY`, one prolix call site, phase 5.

### 6.4 Host orchestration across `aminx.host` / `prolix.api` / proxide — mostly not a collision

The brief expects this to be worst; it is **legitimate layering**, and the author's premise about proxide is false. `prolix/api/ensemble_dispatch.py` wraps `xtrax.tiling.dispatch.make_axis_dispatch`; `aminx/host/plan.py` wraps `xtrax.tiling.BatchPlanner`. Both are thin domain-specific policy layers over xtrax primitives — correct. proxide has no general host-parallel abstraction at all (its only parallelism is `ops/prefetch.py`'s grain autotuning for `create_protein_dataset`; `proxide-parallel-rt` is a ~25-line AtomicUsize registry for WASM). The one genuine crossing, `aminx/host/prep.py` importing `proxide.ops.dataset.create_protein_dataset`, is correct consumption, not reimplementation.

Resolution: **re-scope proxide in the mental model** from "host processing ops" to protein structure I/O + chemistry + physics bridging + a Grain dataset pipeline. `proteinsmc/runner.py` + `sampling/initialization_factory.py` are the same thin-policy shape and stay (as a shim and dissolved-into-kernels respectively), but call xtrax rather than hand-rolling. **Streaming sinks** (5 implementations) are deferred: `io.py` stays hand-rolled minus git-SHA; flag that `proxide/io/streaming/array_record.py` handles the same backend, so the eventual fix may be delete-and-import. `prolix/api/xtc_sink.py`'s documented refusal of `ZarrStagingSink` ("Zarr is wrong for MD traj interchange") is preserved as precedent that a domain-format sink may opt out.

### 6.5 `models/` vs `utils/` "duplication" — there isn't one

Reading all seven same-named pairs refutes the framing. The pattern is 100% consistent: `models/X.py` is config/type declaration (`models/mutation.py` is an 11-line `MutationFn` alias; ~250L total across all seven), `utils/X.py` is implementation (`utils/mutation.py`'s `mutate`/`diversify`/`chunked_mutation_step`; ~1000L total). Nothing on either side is dead — every config is instantiated by `runner.py` or `oed/experiment.py`, every implementation is imported by runner/sampling/scoring. Only the shared submodule *names* across two packages create the appearance.

The one genuine internal redundancy is `models/protocols.py` — an abandoned second contract whose `FitnessFn` takes `dict[str, Array|float|int]` while `models/fitness.py`'s `StackedFitnessFn` takes `PyTree|Array`, referenced only as a TYPE_CHECKING import in `particle_systems/smc.py`. It is deleted and reconciled into `api.py`.

The dissolution in §4 is therefore **colocation-by-concern, not de-duplication**, and it is the most optional item in this plan (§8 Q2). Deprecated re-export shims at `proteinsmc/models/__init__.py` and `proteinsmc/utils/__init__.py` are mandatory for one release — 45 of ~50 test files import from those paths.

### 6.6 Other collisions, briefly

| Collision | Resolution |
|---|---|
| **ESM unowned** (`utils/esm.py` = only implementation in ecosystem; aminx has zero) | → aminx `models/esmc.py`. proteinsmc keeps nothing |
| **Memory-budget estimation** (4 implementations; `estimate_memory_theoretical` defined identically in aminx and prolix planners) | xtrax owns budget + estimator *protocol*; domain byte-estimators stay as caller closures; `proteinsmc/utils/memory.py`'s wall-clock benchmarking deleted; aminx/prolix estimators should at least be renamed to say what they estimate |
| **Length bucketing** (4-way; `LENGTH_BUCKETS=(100,200,400,800,1200)` vs `SEQ_BUCKETS=(200,400,800,1200)`) | xtrax owns `Bucket`/`select_bucket`/`bucketize`; ladders are domain data passed as `AxisSpec`; `proteinsmc/padding.py` deleted (dead) |
| **AA alphabet / MPNN↔AF permutation** (4 vocabularies, 3 permutations) | proxide owns; proteinsmc deletes its protein alphabet entirely (core needs only `q`); the aminx↔proxide byte-identical duplication is fixed **there**, not routed through proteinsmc |
| **Shared type vocabulary** | Unresolved by design. proteinsmc keeps ~30 lines of local aliases rather than importing `proxide.core.types`, because that edge is the one being cut |
| **Per-step `io_callback` wiring** (4 modules + io.py) | → one `xtrax.stages.AxisBoundary` parameter on `drive.sample()` |
| **Dedup-gather** (`oed/smc_tree.py`'s shared-key branch-history dedup) | Moves to asr with the file; should become an `xtrax.tiling.DedupSpec` keyed on branch prefix (`prolix/api/ensemble_dedup.py` is the reference) |
| **Run provenance** (5-way git-SHA capture) | bathos owns. `io.py:create_metadata_file`'s shell-out deleted; `oed/tracking.py` deleted |
| **Research-integrity governance in xtrax** (26 `xtrax/loop/` modules duplicating bathos) | Out of scope, named as a credibility precondition (§8 Q5) |
| **PRNG helpers** (`utils/key_management.py`, tested, imported nowhere) | Deleted, not adopted — adoption changes the PRNG stream for zero gain |
| **NK** (proteinsmc's 3 files vs `trex/nk_model.py`, 292L) | Pick one before phase 3 (§8 Q4). Recommendation: port `smc_tree` onto `proteinsmc.landscapes.nk` — which also supplies the consumer edge phase 3 requires |

---

## 7. Migration sequence

Every phase states what breaks if skipped. Phases −1 through 2 are independently shippable and reversible; phase 3 is a coordinated two-repo commit-pair; phases 4–5 are independently shippable.

### Phase −1 — make the repo buildable (blocking, hours) **[fixes SERIOUS: phase ordering impossible]**

**Change:** drop `"prxteinmpnn"` from `pyproject.toml:20` *(line corrected from `:21` — Addendum C, C9; this is a line-targeted edit)*. It is only reachable behind `importlib.util.find_spec` in `scoring/mpnn.py:14`, so removing it breaks nothing that currently runs.
**Unblocks:** everything. `uv run python -c ...` currently fails at resolution — *"Because prxteinmpnn was not found in the package registry and your project depends on prxteinmpnn, we can conclude that your project's requirements are unsatisfiable"* — and `.venv/lib/*/site-packages` contains only `_virtualenv.pth`. The package is now `aminx` (`~/projects/aminx/pyproject.toml:6`); the stale `src/prxteinmpnn.egg-info` is what asr's bootstrap script patches around.
**Verify:** `uv sync` green; `uv run pytest -q` produces a **captured baseline** (expect failures; record them).
**If skipped:** phase 0's "expect 4 HMC/NUTS failures to become loud" is not a checkable statement, because no test can run.

### Phase 0 — credibility repair (blocking, ~3–5 days, not 1) **[fixes FATAL objection #2]**

**Change, scoped correctly:** *make every registered sampler run end-to-end under the production `get_fitness_function`* — not "fix the `if` branch."

This requires choosing **one** fitness signature first, because there are currently three mutually incompatible ones:
- `utils/fitness.py:114` returns `jnp.concatenate([jnp.array([combined]), all_scores])`, length ≥ 2 always.
- `smc.py:119-124` consumes it **per-sequence** and slices `stacked_fitness[:, 0]`.
- `parallel_replica.py:135-150` consumes it **per-batch and expects a scalar**: with the real stacked fitness, `accept` has shape `(2,)` and `lax.cond` rejects a non-scalar predicate.
- The test suite hides this with a **fourth** contract: `tests/sampling/particle_systems/test_parallel_replica.py:69-73`'s `mock_fitness_fn` returns `jnp.ones((sequences.shape[0],))`, a shape neither `get_fitness_function` nor `smc.py` produces.

So: adopt the batched `Scores` signature, move per-component scores into `KernelInfo.extra`, rewrite `island_migration.migrate` and its fixtures against it, fix `runner.py:303`.
**Verify:** each of the six registered samplers runs ≥3 steps on a 32-particle NK target and produces a non-degenerate `KernelInfo`. The 4 known `test_initialization_factory.py` int8→float32 failures convert from silent no-ops to loud failures — that is the point.
**If skipped:** the repo's central claim stays false and no sibling should depend on it. Everything downstream builds on a lie.

### Phase 1 — free subtraction (hours, independently shippable)

**Change:** delete `padding.py`, `utils/jax_uuid.py`, `utils/key_management.py`, `utils/config_unpacker.py`, `utils/serialization.py` (after §3.5 lands), `models/lineage.py`, `models/types.py`, `models/protocols.py`. Port `NKLandscape`/`InteractionCarry` from `flax.struct` to `eqx.Module`.
**Unblocks:** shrinks everything downstream; makes the flax dependency question honest.
**Verify:** `uv run pytest` no worse than the phase-0 baseline; `grep -rn "jax_uuid\|config_unpacker\|key_management" src/ tests/` empty.
**If skipped:** ~6 dead modules get carried into whatever new boundaries get drawn.

### Phase 2 — additive contract, old surface still working (~1–2 weeks)

**Change:** land `api.py`, `compose.py`, `registry.py`, `drive.py`, `proposals.py`, `schedules.py`, `landscapes/nk.py`, `kernels/` (with `_blackjax.py`), `_xtrax.py`. Convert `_run_smc_chunk` and `_run_prsmc_chunk` from `jax.jit(static_argnames=("fitness_fn",...))` to **`eqx.filter_jit`**. Switch `io.py` to msgpack + kernel skeletons. Keep `runner.run_experiment` and the `proteinsmc.models` / `proteinsmc.utils` shims working throughout.

**Critically, also in this phase:** land `proposals.PointMutation` and the `chunked_map -> xtrax.transforms.safe_map` swap **with both surfaces present**, and repoint `oed/smc_tree.py:13-14` onto them. Verify the divisibility constraint first — `xtrax/transforms/map.py` raises `ValueError` when `n % batch_size != 0` whereas `lax.map(batch_size=)` does not, and `smc_tree` defaults to `inference_batch_size=64` against arbitrary `pop_size`.

**Unblocks:** phase 3 becomes a *single* repoint for asr instead of two.
**Verify:** the §3.6 target call site runs with an in-repo `NKFitness`; a synthetic `eqx.Module` scorer carrying `jax.Array` weights passes through `SMCKernel.step` under `filter_jit` (this is the direct test for FATAL #1); `runner.run_experiment` still passes its existing tests.
**If skipped:** aminx's scorers cannot cross the jit boundary at all, and asr breaks twice.

### Phase 3 — the two extractions (one commit-pair; breaks asr at 10 sites) **[gated — fixes FATAL objection #4]**

**Gate, mandatory:** phase 3 does not start until at least one **load-bearing** consumer edge exists, defined by the acceptance test *"a sibling repo's test suite fails if proteinsmc is uninstalled."* The cheapest real edge is already in the code: `oed/smc_tree.py:98-103` hand-rolls multinomial resampling — port it onto `psmc.kernels.SMC` before the move, so asr keeps a genuine runtime dependency. Second cheapest: aminx's `MPNNFitness` subclassing `psmc.Fitness` (a real import, not a TOML string). **Without this gate, phase 3 removes everything anyone can run end-to-end and buys a dependency edge made of strings** — asr's entire current import surface points at code this phase moves out.

**Change:**
- `oed/` (all 9 live modules incl. `smc_tree.py`) → asr, merging into the existing `asr/src/asr/oed/`. Reconcile against asr's already-adapted `gp.py` (ARD kernel, masked static shapes) and `structs.py` (`ASRDesign`) — do not overwrite. `oed/tracking.py` → bathos.
- `utils/esm.py` + `scoring/esm.py` + `scoring/mpnn.py` + the ESM constants block → aminx. Delete (not rename) the `find_spec("prxteinmpnn")` gate. Land the two `[project.entry-points."proteinsmc.fitness"]` entries.
- Retire asr's `bootstrap_vendor_proteinsmc.sh` rewrite hack.

**Verify:** asr's 10 import sites (the 8 known plus `smc_tree.py:13-14`'s two, plus `oed/nk.py:5`'s `NKLandscape`) resolve; `grep -rn "aminx\|prxteinmpnn\|proxide" src/proteinsmc/` empty; `FITNESS["mpnn"]` resolves from an installed aminx and raises a clear error without one.
**If skipped:** the dependency edge never flips; proteinsmc stays a leaf consumer of a stale package name.

### Phase 4 — xtrax adoption (independently shippable)

**Change:** delete `utils/memory.py`, `utils/pmap_utils.py`, `jax_utils.chunked_map`; write the ~150-line `_xtrax.py` adapter modeled on `prolix/tiling/xtrax_adapter.py`. Replace the four `io_callback` params with one `AxisBoundary`. **Set `requires-python = ">=3.13"`** — `~/projects/xtrax/pyproject.toml:6` requires 3.13 and `xtrax/transforms/scan.py:6` uses PEP-695 syntax (`def safe_scan[Carry, X, Y](...)`) that will not parse on 3.11. This is a two-minor-version bump from `pyproject.toml:9`'s current `>=3.11` and is a real cost of the pin (§8 Q5).
**Verify:** `xtrax plan` reproduces the batch sizes the deleted heuristics chose, or better; ordered-IO streaming works under `Tap`.
**If skipped:** proteinsmc keeps four private reimplementations of xtrax's boundary layer.

### Phase 5 — optional ecosystem convergence

**Change:** the three-way PT parity harness (§6.3) → `kernels/replica_exchange.py` only if it passes; prolix ladders into `schedules.py`; aminx `SAMPLERS -> DECODERS`; ship `decoder_proposal` + ADRs; rewrite `ste.py` on optax and register it.
**Verify:** parity tests against each sibling's current implementation output.
**If skipped:** the ecosystem-wide duplication remains standing — which is the honest status quo, not a regression.

### Phase 6 — deletions of the shims

Remove `runner.py`, `proteinsmc/models/__init__.py`, `proteinsmc/utils/__init__.py` one release after phase 3.

---

## 8. Open questions for the author

**Q1. Is proteinsmc's product the contract or the kernels?**
*Recommendation:* the contract. Write it as the README's first paragraph so the claim is falsifiable, with the success criterion "≥2 independent implementors of `psmc.Fitness`, one in-repo (NK) and one out-of-repo (aminx)" testable at phase-3 exit.
*Tradeoff:* selling the contract means admitting that 633 of ~1,700 sampling LOC is blackjax delegation and that the repo is structurally a Protocol + drive loop over an existing library. That is a smaller claim, but it is the one that survives.

**Q2. Should `models/` and `utils/` actually be dissolved?**
*Recommendation:* yes, but with mandatory deprecated re-export shims for one release, and only after phase 2 lands.
*Tradeoff:* dissolution buys coherence, not correctness. It is the single largest source of churn — it breaks every `from proteinsmc.models import ...` path and every test that constructs a config, and 45 of ~50 test files do exactly that. A fully defensible variant of this plan keeps both packages and changes only what the edge flip requires. **This is the most optional item in the table; decide it explicitly rather than letting it happen.**

**Q3. Force or merely permit aminx/prolix adoption of shared sampling kernels?**
*Recommendation:* permit, and gate on the parity harness (§6.3).
*Tradeoff:* permitting means the ecosystem-wide duplication the exercise was chartered to find is still standing after this change, and "possible later" has a poor record. There is also an unmeasured performance question: aminx's Potts energy is currently inlined so the compiler sees `_potts_pair_energy` directly; routing through an opaque `Fitness` puts a call boundary on an `(n,n,q,q)` coupling tensor. Someone must benchmark that, and you must decide whether an aminx fused fast path alongside the generic kernel is an acceptable outcome or a failure.

**Q4. proteinsmc's NK or trex's NK?**
*Recommendation:* proteinsmc's, and port `smc_tree.py` onto `proteinsmc.landscapes.nk` in phase 2 — which conveniently supplies the phase-3 consumer gate.
*Tradeoff:* today `smc_tree.py:8` imports `from trex import nk_model` while `proteinsmc/utils/nk_landscape.py` (239L) sits unused by the only live consumer, and `trex/nk_model.py` is 292L. Post-plan, asr's dependency closure contains both. If you prefer to concede NK to trex, then `proteinsmc/landscapes/nk.py` must be deleted, the "reference implementation of the protocol" claim retracted, and replaced with a genuinely dependency-free toy target (Hamming or Ising, ~30 lines) whose only job is to demonstrate `Fitness`. Either is defensible; shipping both is not.

**Q5. Do you accept a hard `xtrax>=0.4.0a5,<0.5` pin — including the Python 3.13 floor?**
*Recommendation:* yes, but only after xtrax narrows.
*Tradeoff:* the pin couples the sampler's release cadence to an alpha library **and** bumps `requires-python` from 3.11 to 3.13. xtrax also currently ships 26 research-integrity governance modules (`xtrax/loop/`: `prereg_match`, `attestation_evidence_gate`, `seed_gate`, `campaign_approval_gate`, `sidecar_drift_gate`, `promotion_gate`, `metrics_provenance`, `multi_metric_ratchet`, …) plus `xtrax/composition`, duplicating bathos's entire domain inside what is supposed to be a JAX kernel library, wired onto a feature whose own docstrings (#3059/#3067/#2181) say does not exist. Three repos are about to hard-pin it. Extracting `xtrax/loop` to bathos is a precondition for xtrax being a credible narrow base — but it is a different repo and arguably a different project. If you will not narrow xtrax, the honest alternative is for proteinsmc to keep a small private tiling shim rather than pin a wide alpha, which loses most of phase 4's win.

**Q6. Who builds the sequence→structure bridge, and when?**
*Recommendation:* aminx (`ebm/structure_prediction.py` is the nearest existing thing), and not as part of this migration.
*Tradeoff:* until it exists, the `PhysicsFitness` case in §3.4 is a well-typed contract with no working instance — a prolix physics energy cannot be a proteinsmc fitness function today, no matter how good the protocol is. **Do not claim otherwise in the docs.** The contract is written so the bridge slots in as an injected `fold` callable with no repo boundary moving, but that is a promise, not a feature.

**Q7. Who fixes asr, and when?**
*Recommendation:* schedule it as a coordinated two-repo commit alongside phase 3, with you present.
*Tradeoff:* phase 3 breaks asr at 10 call sites and requires reconciling asr's already-adapted `oed/gp.py` and `oed/structs.py` against the incoming originals. Because asr vendors proteinsmc as a uv workspace member with a bootstrap-patch script, the change must land in the vendor pin and the source repo together. That is not a mechanical rename an agent can do unattended.

**Q8. HMC and NUTS: relax, delete, or unregister?**
*Recommendation:* relax (explicit `relax:` boundary), keep HMC registered, keep NUTS unregistered pending a parity test.
*Tradeoff:* `nuts.py`'s own docstring calls itself "a conceptual placeholder for demonstration purposes." Whether gradient-based sampling on a discrete alphabet is a direction you actually want, or two kernels shipped because blackjax made them cheap, is a question only you can answer. Deleting both is cheaper and loses nothing that currently works.

---

## 9. Risks

**R1 — The intermediate state is worse than the status quo (highest).** Phase 3 removes everything a user can run end-to-end today; if the aminx adapter work does not land promptly after phase 2, you are left with an elegant unproven core and nothing plugged in. *Mitigation:* the phase-3 consumer gate (§7) is non-negotiable; phase 2 is strictly additive with the old surface working throughout, so the plan is safe to abandon at the end of phase 2; phase 3 is a single commit-pair across two repos.

**R2 — Phase 0 is much larger than "fix an `if`".** Three incompatible fitness signatures plus a test suite encoding a fourth means the dispatch fix is a contract-selection exercise touching `parallel_replica.migrate`, its fixtures, and 45 of ~50 test files. *Mitigation:* budget days not hours; capture a phase −1 baseline; do the signature choice before writing any kernel code, so the choice is made once.

**R3 — The jit boundary conversion has silent recompilation semantics.** Under `jax.jit(static_argnames=("fitness_fn",))` a new scorer instance is a new hash → full recompile. Under `eqx.filter_jit` it is a shape/treedef match → no recompile, and a scorer whose *static* config differs but whose treedef matches will silently reuse the wrong trace if fields are misclassified. *Mitigation:* every `Fitness` subclass must classify non-array config as `eqx.field(static=True)` explicitly; add a test that two scorers differing only in a static field produce different traces.

**R4 — Existing output files become unreadable.** `io.py` has no versioning, and both the skeleton builder and the config it depends on are deleted. *Mitigation:* land the msgpack switch and the JSON header in phase 2, before deleting `utils/serialization.py`; state the break in the changelog; if any archived run matters, convert it before phase 1.

**R5 — The `chunked_map -> safe_map` swap changes failure behavior.** `xtrax/transforms/map.py` raises `ValueError` on `n % batch_size != 0`; `lax.map(batch_size=)` does not. `smc_tree`'s `inference_batch_size=64` against arbitrary `pop_size` can start failing. *Mitigation:* audit call-site population sizes in phase 2; either round populations to the bucket or wrap with padding.

**R6 — Flax pytree identity split across the repo boundary.** `NKLandscape` is consumed on both sides (`scoring/nk.py` stays, `oed/nk.py:5` moves to asr); if asr redefines rather than imports it, the two are distinct registered pytree types that will not unify across a `scan`/`cond` boundary. *Mitigation:* port to `eqx.Module` in phase 1; make "asr imports, does not redefine" a phase-3 acceptance criterion.

**R7 — Entry-point discovery is import-time magic.** `FITNESS["mpnn"]` resolves through installed-package metadata, so a missing aminx becomes a `KeyError` at config time rather than an `ImportError` at module load. *Mitigation:* clear error text naming the package and extra to install; ship `proteinsmc.fitness.available()` and a `proteinsmc scorers` CLI verb.

**R8 — Phase 5's Potts convergence regresses aminx performance.** Routing an inlined `(n,n,q,q)` coupling energy through an opaque `Fitness` adds a call boundary the compiler currently sees through. *Mitigation:* benchmark before committing; if it regresses, aminx keeps a fused fast path alongside the generic kernel with a parity test — the same discipline it already runs for EPIC #1541.

**R9 — The PT parity harness fails and the "unification" story collapses publicly.** Given §6.3, this is more likely than not. *Mitigation:* this is already priced in — the shared kernel is conditional, `island_migration.py` is named honestly, and a negative result is a publishable finding, not a failure of the migration.

**R10 — Nobody retires the shims.** `runner.py` and the `models`/`utils` re-export packages outlive their one release and become permanent, leaving two public surfaces forever. *Mitigation:* put the deletion in phase 6 with a dated `DeprecationWarning` that names the removal release, and file it as a bathos-tracked task at the moment the shim is written.
---

## Addendum A — aminx decomposition (added 260813, post-review; **revised** after author challenge)

*Author-directed follow-up to §1's third broken assignment ("aminx has expanded from running models into running dynamics on models"). Measured against the aminx source.*

> **Revision note.** A first draft of this addendum recommended extracting **both** `ebm/` and `potts/`. The `potts/` recommendation was **wrong** and is withdrawn in A.4 — `potts/` is PottsMPNN, an MPNN-family model. The error came from inferring domain separation out of low import counts plus an ADR that actually separates on *execution pattern*. The `ebm/` recommendation was re-verified against the same failure mode and stands. A.7 records the consequence for §1.

### A.1 Correction to §1: `ebm/langevin.py` is NOT molecular dynamics

**§1 claims `aminx/ebm/langevin.py` is "an MD integrator over backbone coordinates (prolix's domain)". This is wrong and must not be acted on.**

`ebm/langevin.py`'s module docstring: *"Fixed-noise-level Langevin/reverse-SDE equilibration ... Ported from `~/repos/ProteinEBM/protein_ebm/scripts/run_dynamics.py` ... and `r3_diffuser.py` (`diffusion_coef`/`drift_coef`)"*. Every step reads `model.aux_score`/`model.energy` — a learned, non-conservative neural surrogate head. A grep for `force_field|forcefield|amber|gbsa|bonded|lennard|coulomb` returns **0 hits** in both `ebm/langevin.py` and `ebm/diffusion.py`.

Score-based generative sampling of a diffusion model, not force-field dynamics. prolix owns AMBER/GBSA/bonded physics. Moving `ebm/langevin.py` to prolix is a category error on a shared word.

The §1 *observation* stands — aminx does host more than "running models" — but the remedy is extraction into its own package (A.3), not relocation into prolix.

### A.2 Measured structure

aminx is ~42,567 LOC. It hosts the MPNN model family (in several architectural variants), one unrelated model family, and shared infrastructure.

| Subsystem | LOC | Files | Inbound edges | Verdict |
|---|---|---|---|---|
| `ebm/` — ProteinEBM diffusion transformer | 6,608 | 16 | **0** | **extract** |
| `potts/` — PottsMPNN + TRW inference | 2,990 | 13 | 2 (root `cli.py`) | **stays** — MPNN-family (A.4) |
| MPNN core + infra (`model`,`inference`,`types`,`host`,`sampling`,`scoring`,`tiling`,`io`,`run`,`training`,`utils`,`parity`,`profiling`) | ~33k | — | dense mesh (`inference→types` 53, `host→utils` 25, `host→types` 20, `host→inference` 14, `host→run` 14) | keep as aminx |

**Import counts alone are not evidence of separability** — see A.4. Architectural lineage has to be read from the source.

### A.3 Extraction — `ebm/` (the only one)

Zero inbound edges. Total coupling into aminx is **seven imports**:

```
ebm/plan.py:65,66              aminx.utils.safe_map / safe_scan     -> xtrax.transforms (per §6.6)
ebm/langevin_schedule.py:201   aminx.utils.safe_scan                -> xtrax.transforms
ebm/ddg_stability.py:114       aminx.utils.aa_convert.MPNN_ALPHABET -> proxide (per §6.6)
ebm/conformational_biasing.py:78   aminx.utils.aa_convert.AF_ALPHABET -> proxide
ebm/conformational_biasing.py:273  aminx.io.parsing.parse_structure   -> proxide (already a deferred wrapper)
ebm/trunk.py:113               aminx.model.diffusion_mpnn.SwiGLU    -> residual
```

After this plan's own normalizations, **residual coupling is one import: a `SwiGLU` MLP block**. Copy it or promote it to a shared layers module.

**Architecture check (the test `potts/` failed).** `ebm/` is a Boltz-1-style DiffusionTransformer, mechanically ported from `~/repos/ProteinEBM/protein_ebm/model/layers.py` (`AdaLN`, `AttentionPairBias`, `DiffusionTransformerLayer`, `FourierEmbedding`, `RelativePositionEncoder`). It is **not** MPNN-architecture. Its MPNN contact is limited to:
- reusing the `SwiGLU` MLP block verbatim (`trunk.py:273`, "reused, not reimplemented");
- the `MPNN_ALPHABET` amino-acid indexing convention (`ddg_stability.py`) — a vocabulary §6.6 already assigns to proxide.

`trunk.py:309` explicitly records that `AttentionPairBias` is **not** a reuse of `model.diffusion_mpnn`'s, and `trunk.py:62` distinguishes its `FourierEmbedding` from that module's `SinusoidalEmbedding`. Shared vocabulary and one MLP block — not shared architecture.

Supporting evidence the seam is intentional:
- `ebm/plan.py` documents a deliberate refusal to depend on `aminx.host.plan`/`aminx.tiling.axes` (design spec §3.2/§10, BLOCKER-2), citing an incompatible `EncoderOutput` shape.
- `ebm/` ships its own `dispatch.py` (233L), `plan.py` (314L), `contracts.py` (73L) — a parallel infrastructure stack.
- `ebm/langevin.py` names `xtrax` `CarrySpec`+`Scan` as the intended outer-loop mechanism, so the extracted package's direction is already `ebm -> {xtrax, proxide}`, consistent with §2's DAG.

Contents: a self-contained model family with its own training and applications — `model.py`, `trunk.py`, `readout.py`, `diffusion.py`, `training.py`, `checkpoint.py`, plus `structure_prediction.py`, `ddg_stability.py`, `decoy_ranking.py`, `conformational_biasing.py`.

### A.4 `potts/` STAYS in aminx — **[CORRECTED: it is PottsMPNN]**

**A prior draft recommended extracting `potts/`. That was wrong.**

`potts/model.py:72` — `PottsModel` is *"Potts MPNN with TRW inference head on k-NN geometric graphs."* It is a member of the MPNN family:

- **Runtime dependency on aminx's MPNN featurizer.** `runner.py:188-190` — *"Step 3.5: Compute k-NN graph features via ProteinFeatures"* — instantiates `ProteinFeatures` from `aminx.model`. Not a type import; the model's featurization step. `model.py:75` confirms it "builds k-NN edges from structure coordinates using ProteinFeatures".
- **Identity alphabet alignment.** `model.py:11` — *"Potts uses the canonical MPNN alphabet (q=21, identity mapping)"*, per ADR `260605_potts-alphabet-alignment`. `POTTS_TO_MPNN_ALPHABET_MAP` is a permutation kept for interop, not evidence of a foreign vocabulary.
- **Weight lineage.** `h`/`J` carry a factor-of-2 "from the directed-slot PottsMPNN convention ... preserved to maintain numerical consistency with weight recapture from mistypotts"; checkpoints arrive via `pottsmpnn_to_eqx.py`.
- **Reference architecture** is Birnbaum & Keating, *"Beyond native sequence recovery"* (bioRxiv 2026) — the PottsMPNN paper.

**The ADR wall was misread.** `260605_potts-parallel-not-stageset` states *"PottsModel is a parallel model family (NOT a StageSet consumer)"*. The banned-api rules wall potts off from `aminx.inference.decode` / `aminx.host.plan` / `aminx.types.stages` — the **autoregressive decode pipeline** — because Potts inference is parallel one-shot TRW rather than sequential decode. That is an **execution-pattern** boundary, not a **domain** boundary. potts is an MPNN-family model that does not use the decode machinery.

**What does move is unchanged from §6.1 and needs no extraction:** `potts/sampling.py` (618L) is generic discrete MCMC hardcoded to Potts energies — `gibbs_sweep`, `dlmc_sweep` (Zanella locally-balanced), `parallel_tempering`, `_parallel_tempering_exchange`. Split into (a) proteinsmc kernels and (b) a Potts `Fitness`, it still supplies the second independent `Fitness` implementor for the §7 phase-3 consumer gate. The model, featurizer, TRW head, calibration, PoE, and designer all stay in aminx.

### A.5 Interaction with the main plan

- Does not disturb the §2 DAG. One new leaf package depending on `{xtrax, proxide}`.
- Does **not** supersede §6.1's phase-5 treatment of Gibbs/DLMC/PT convergence — with `potts/` staying put, that item is unchanged: still a same-repo boundary-drawing exercise, still optional, still gated on the §6.3 parity harness.
- **Does not resolve §6.3.** `potts`' `_parallel_tempering_exchange` still runs both parities in one call via a `nonlocal`-mutating closure, so it still does not fit the withdrawn signature.
- §8 Q3's unmeasured performance question (routing an inlined `(n,n,q,q)` coupling energy through an opaque `Fitness`) is **unchanged in scope** — it still concerns aminx's main body.

### A.6 Open question for the author

**Q9. Extract `ebm/` to its own repo, or keep it as an aminx extra?** *Recommendation:* extract — its independence is real, documented, and already enforced by its own parallel infrastructure; it has zero consumers inside its host repo. *Tradeoff:* §8 Q2's maintenance-capacity argument applies; the ecosystem is already five repos and prolix shows what strain looks like. An extra is the cheaper reversible step if you want to defer the repo decision.

### A.7 Consequence for §1's "aminx structurally cannot host the contract" argument

§1 argues aminx *"has already ruled by lint that a non-StageSet iterative sampler cannot live in its main body."* **A.4 weakens this.** `potts/` is precisely a non-StageSet iterative sampler living in aminx's main body; the lint forbids it from importing the decode pipeline, not from existing there. The claim proves aminx walls things off from `inference.decode`, not from the repo.

The §1 *conclusion* — proteinsmc owns the contract — does not depend solely on that argument and is not withdrawn. The surviving grounds are: (a) the contract must span prolix physics energies and NK landscapes, neither of which aminx should reach for; (b) the §2 DAG needs `aminx -> proteinsmc`, and hosting the contract in aminx inverts it; (c) aminx is ~42.5k LOC and adding a cross-ecosystem protocol to it worsens the mass problem this addendum is trying to relieve. **Restate §1's justification on those three grounds and drop the lint argument.**
