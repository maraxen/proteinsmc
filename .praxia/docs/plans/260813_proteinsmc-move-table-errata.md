# Addendum B — Errata to the proteinsmc Ecosystem Partition Plan

**Source plan:** `/home/marielle/projects/proteinsmc/.praxia/docs/plans/260813_proteinsmc-ecosystem-partition.md`
**Audit scope:** all six move groups (aminx, asr, DELETE, xtrax, proxide/negatives, contract/codon), two-pass (verify + adversarial challenge).
**Rule applied:** where verifier and challenger disagree, the challenger's evidenced position wins unless the verifier's evidence is concretely stronger. Every such case is flagged inline.

---

## B.1 Summary

**60 claims audited across six groups. 17 survived unchanged. 14 were refuted. 27 need qualification. 2 are unverifiable without a working environment.**

> **Ledger caveat — added by Addendum C.** Those four counts sum to 60, and 14+27=41 is the "two thirds" figure B.7 quotes. But **they are not reconstructible from this document's own structure**: §B.2 carries 17 numbered entries against a claimed 14 refuted; §B.3 carries 17 numbered entries plus a 6-row table (=23) against a claimed 27; the 17 "survived" claims are never listed anywhere; and the "72 raw claims" figure from which 60 was deduped appears nowhere in this file. The dedup rule was never stated. **Treat 17/14/27 as a rhetorical summary of severity, not as an auditable ledger.** Each finding below stands on its own cited evidence; the aggregate does not.

**The single most dangerous finding is a live, silent cross-repo correctness bug that the move table would carry into a second repo rather than fix.**

proteinsmc's amino-acid integer encoding is the **AlphaFold** ordering — `restypes = ARNDCQEGHILKMFPSTWYV` (`proteinsmc/src/proteinsmc/utils/constants.py:10-31`), aliased as `AA_CHAR_TO_INT_MAP = restype_order` (`:113`). Three things named after ProteinMPNN are built on top of it:

- `PROTEINMPNN_X_INT = 21` (`constants.py:115`) — X is index **20** in the real MPNN alphabet (`proxide/src/proxide/io/parsing/mappings.py:23`, `aminx/src/aminx/utils/aa_convert.py:16`) and **20** in proxide's AF-with-X (`proxide/src/proxide/chem/residues.py:646`). 21 matches neither.
- `PROTEINMPNN_TO_ESM_AA_MAP_JAX` (`constants.py:272-282`) — an AF→ESM map wearing an MPNN name, allocated at length 20 while sequences legally carry 21, so `utils/esm.py:52`'s gather **clamps a stop/unknown residue to index 19 = Valine**.
- `scoring/mpnn.py` performs **no alphabet conversion at all** (`rg 'af_to_mpnn|mpnn_to_af|aa_convert' proteinsmc/src proteinsmc/tests` → exit 1, zero hits) yet feeds these AF-ordered ints to an MPNN-ordered scorer, while aminx applies `af_to_mpnn(...)  # AF -> model space` at four separate host sites before any model call (`aminx/src/aminx/host/runner.py:959,1020,1237`; `host/_sampling_helper.py:581`).

> **Mechanism corrected by Addendum C.** The four `af_to_mpnn` sites are all in `aminx/host/`, and **the live asr path does not go through `aminx/host/` at all** — `asr/src/asr/analysis/biophysics.py:18` calls `aminx.scoring.score.make_score_fn(self.mpnn_model)` directly and `:66-70` passes `sequence` unconverted. So the defect is not "proteinsmc's AF ints meet aminx's MPNN-guarded entry point"; it is **one integer array with two consumers, neither of which converts.** Whichever ordering `ancestor_seq` is actually in, exactly one of the two score streams is scrambled. This relocates the fix: it belongs at the asr call site as much as in proteinsmc's constants, and G1 as originally written covered neither (see Addendum C, C1/N2).

This is already live in asr. `asr/src/asr/analysis/biophysics.py:89` passes one `ancestor_seq` to both `compute_esmc_site_scores` (→ proteinsmc's AF table) and `compute_mpnn_site_scores` (→ aminx's MPNN scorer). `asr/tests/spikes/test_esmc_proteinsmc_spike.py:18-20` documents the alphabet as `A=0, C=1, D=2, ... V=17, W=18, Y=19` — MPNN order — and builds the KRAS peptide accordingly; decoded under the AF table that actually indexes it, `MTEYKLVVVGAGGVGKSALTIQLI` becomes `LTDVHIWWWQAQQWQHSAITGFIG`. **Five of the 24 positions coincide** — corrected from "one" by Addendum C: the MPNN→AF permutation has exactly three fixed points (A→A at index 0, S→S at 15, T→T at 16), and this peptide contains A twice (positions 11, 18), S once (17) and T twice (2, 20). The decoded string itself was independently recomputed and is exact. The plan's §4 routes `scoring/mpnn.py` and `scoring/esm.py` into aminx and DELETEs the alphabet, which relocates the defect into the repo whose `aa_convert.py` exists specifically because those two orderings differ.

Second-order but structural: **the plan's foundational architectural premise is the potts error verbatim.** Plan line 9 argues aminx "structurally *cannot* host" an iterative sampler because `aminx/pyproject.toml:143-169` walls off `potts` by lint. That block contains **sixteen** entries, of which only four are the potts wall (`aminx.inference.decode`, `aminx.host.plan`, `aminx.types.stages`, `aminx.inference.logits`), each with the message "PottsModel is parallel, not a stageset consumer"; the other twelve are unrelated `xtrax.*` public-API-only rules. *(Count corrected from "exactly four modules" by Addendum C — the conclusion is unaffected, the description of the artifact was wrong.)* It bans a *pipeline*, not a *pattern*. aminx already ships `aminx/src/aminx/potts/sampling.py` — 618 LOC, docstring line 1: "Single-site Gibbs updates and parallel tempering for pairwise Potts models" — with `gibbs_sweep`:195, `_parallel_tempering_exchange`:289, `parallel_tempering`:491. That is a direct collision with `proteinsmc/src/proteinsmc/sampling/particle_systems/parallel_replica.py:50-58 migrate` and `proteinsmc/src/proteinsmc/sampling/gibbs.py:73-79`, which the plan explicitly denies at line 9.

Third: **five DELETE rows target live code.** `utils/memory.py` and `models/memory.py` are on `run_experiment`'s default path (`runner.py:36,112,131`; `models/memory.py:47 enable_auto_tuning: bool = field(default=True)`). `models/types.py` has 15+ importers. `sampling/initialization_factory.py` is 393 lines of which the string dispatch the plan cites is ~30. `oed/tracking.py`'s checkpoint and ArrayRecord-index halves have no destination in bathos.

Nothing about the plan's *direction* is wrong. The DAG, the codon quarantine's placement decision, the xtrax dependency direction, and the "asr owns the OED harness" conclusion all survive. What does not survive is the **move table as a set of executable instructions** — roughly a third of its rows are wrong in a way that would produce a silent failure rather than a loud one.

---

## B.2 Refuted claims

### B.2.1 `utils/constants.py` ESM/alphabet section → aminx "next to `aa_convert.py`" *(destructive)*

**Plan claims** (§4, line 302/307): the ESM constants including `PROTEINMPNN_TO_ESM_AA_MAP_JAX` move to aminx beside `aminx/utils/aa_convert.py`; `AA_CHAR_TO_INT_MAP`, `restypes`, `PROTEINMPNN_X_INT` are DELETEd because "canonical ordering already lives at `proxide/chem/residues.py`" and "the core needs only `q: int`."

**Actually true.** Four separate defects:

1. `PROTEINMPNN_TO_ESM_AA_MAP_JAX` is an **AF→ESM** map. Constructed at `constants.py:272-282` by iterating `AA_CHAR_TO_INT_MAP` = `restype_order` = `restypes` (`constants.py:10-31`), which is character-for-character `aminx/src/aminx/utils/aa_convert.py:17`'s `AF_ALPHABET[:20]`, not `:16`'s `MPNN_ALPHABET`. *(Line numbers corrected by Addendum C — this section had them transposed and contradicted B.1, which cites `:16` for `MPNN_ALPHABET` correctly.)* Filing it beside `aa_convert.py` places a mislabelled permutation next to the two named permutations that refute it, in a package where `ProteinSequence` is MPNN-ordered by construction (`aa_convert.py:96-101` routes every string through `af_to_mpnn`).
2. **Length-20 array, 21-valued domain.** `constants.py:272-276` sizes it `AMINO_ACIDS_NUM_STATES = 20` (`:119`, corrected from `:118` by Addendum C); `PROTEINMPNN_X_INT = 21` (`:115`) is a legal sequence value; `utils/esm.py:52` is a JAX gather, which clamps → every X/stop scores as index 19 = Valine.
3. **Two dead branches.** `constants.py:290` and `:294` test `"X" in AA_CHAR_TO_INT_MAP`; `restype_order` has 20 entries and no `"X"` key (only `restype_order_with_x` at `:37` does). Both branches are unreachable.
4. **The move is mechanically impossible as scoped.** The map is derived from `AA_CHAR_TO_INT_MAP` and `AMINO_ACIDS_NUM_STATES` at import time. The same table row DELETEs both.

And `PROTEINMPNN_X_INT` cannot be deleted with the alphabet at all: it is the stop/unknown sentinel of the codon path the plan **keeps**. Read at `utils/translation.py:73,95`, `utils/mutation.py:76`, `scoring/cai.py:65`; it is the fill value of `CODON_INT_TO_RES_INT_JAX` (`constants.py:120-124`). Four tests assert its literal value (`tests/scoring/test_cai.py:11,64,91`; `tests/utils/test_translation.py:5,40,61,72`; `tests/utils/test_mutation.py:10`; `tests/utils/test_constants.py:40`).

Nor is `AA_CHAR_TO_INT_MAP` a leaf duplicate — it is the **constructor** of three kept tables: `CODON_INT_TO_RES_INT_JAX` (`constants.py:128-129`), `ECOLI_MAX_FREQS_JAX_list` (`:211-212`, sized `len(AA_CHAR_TO_INT_MAP)+1`), and the ESM map (`:279-283`).

> **Verifier vs challenger:** the verifier CONFIRMED the `restypes`/`AA_CHAR_TO_INT_MAP` delete on ordering-equivalence alone, running no consumer search and never mentioning asr. **Challenger wins** — it ran the consumer search, found the constructor role and the live asr breakage, and surfaced the internal contradiction below.

**Internal contradiction the plan must resolve:** if the ordering is deleted, proteinsmc must import it from proxide to rebuild those three tables. But §2's P3 celebrates *eliminating* the single proteinsmc→proxide edge, and §2's P2 argues proteinsmc must not import proxide because it drags a compiled `_proxider.abi3.so` into a jax-only L1 leaf (`ls proxide/src/proxide/` confirms the `.so`). Three plan positions that are individually defensible and jointly inconsistent.

**Corrected move-table entries:**

| File / symbol | Corrected action |
|---|---|
| `constants.py` ESM vocab (`ESM_SEQUENCE_VOCAB`, `ESM_AA_CHAR_TO_INT_MAP`, `ESM_BOS/EOS/PAD/UNK/MASK_ID`) | → `aminx/model/esmc.py` (model-intrinsic; note **`model/` singular** — `ls aminx/src/aminx/` shows no `models/`) |
| `PROTEINMPNN_TO_ESM_AA_MAP_JAX` | **REBUILD, do not move.** Rename to state the source alphabet; size 21; explicit `X → <unk>` replacing the two dead branches at `constants.py:290-297`. ⚠ **Addendum C, C3:** the original text said "rederive from `aminx.utils.aa_convert.MPNN_ALPHABET`", which contradicts G1's own assumption that proteinsmc sequences stay **AF**-ordered. If they stay AF, this table's *content is already correct* and rebuilding it from `MPNN_ALPHABET` would break a working table — only the name, the length, and the dead branches are defective. **Rederive from whichever alphabet gate G-α declares canonical, not from `MPNN_ALPHABET` unconditionally.** |
| `PROTEINMPNN_X_INT` / `STOP_INT` / `UNKNOWN_AA_INT` | → `proteinsmc/codon/tables.py` with the codon split. Resolve 21-vs-20 as an explicit, tested decision. **Blocks** the `restypes` delete |
| `restypes` / `AA_CHAR_TO_INT_MAP` / `AMINO_ACIDS_NUM_STATES` | DELETE **only after** the three derived tables above are rebuilt against an explicit source. Repointing at `proxide/chem/residues.py:646` flips the X sentinel 21→20 for nine consumers at once — a semantic change the plan books as a pure deletion |

---

### B.2.2 `oed/experiment.py` → asr "asr already has this file; merge" *(destructive)*

**Plan claims:** merge into `asr/src/asr/oed/experiment.py`.

**Actually true.** Filename identity, nothing else. proteinsmc's (179 LOC) builds an `SMCConfig`, calls `proteinsmc.runner.run_experiment`, reads the run's ArrayRecord back via `proteinsmc.io.read_lineage_data_range` + `oed.tracking.get_run_records_range`, and computes divergence metrics (`experiment.py:111-177`). asr's (308 LOC) does bucketed static-shape padding (`get_n_bucket`/`get_k_bucket`/`get_q_bucket`), builds a **trex** NK landscape dict, and calls `trex.evals.benchmark.run_trex_landscape_aware_configurable` (`asr/src/asr/oed/experiment.py:69-211`). No shared function name; no shared import beyond JAX.

asr's file is load-bearing: imported as `run_asr_experiment_padded` by `asr/scripts/oed_exploration.py:27`, `scripts/grid_search_oed.py:21`, `scripts/controlled_sweep.py:25`, `tests/test_oed_integration.py:8`, and it is the sole in-`src` consumer of `asr/oed/padding.py` (`experiment.py:19-24`). Verified importable: `asr/.venv/bin/python -c "import asr.oed.experiment"` → OK.

**Corrected entry:** `oed/experiment.py` → `asr/oed/smc_experiment.py`. **The plan must state explicitly that `asr/oed/experiment.py` is never overwritten.** Its own only caller is `oed/run.py:12,215,248`.

---

### B.2.3 `oed/structs.py` → asr "asr has `ASRDesign`" *(high)*

**Plan claims:** merge into `asr/src/asr/oed/structs.py`.

**Actually true.** The two struct pairs describe different experiments. `OEDDesign` (`proteinsmc/src/proteinsmc/oed/structs.py:20-35`) parameterizes an SMC sampler run: `population_size`, `n_generations`, `diversification_ratio`. `ASRDesign` (`asr/src/asr/oed/structs.py:31-52`) parameterizes a phylogenetic ASR experiment: `n_leaves`, `lambda_val`, `mutation_rate_noise_std`, `use_smc`. `OEDPredictedVariables` (`:38-46`) and `ASRPredictedVariables` (asr `:55-76`) share **zero** fields. Only `OEDFeatureMode` is identical (proteinsmc `:13-17` == asr `:10-14`).

Both are `flax.struct.dataclass`, so merging registers two incompatible pytree types in one module, and every downstream module is hard-bound to one: `oed/opt.py:45` hardcodes `["N","K","q","population_size","mutation_rate","diversification_ratio"]` (three absent from `ASRDesign`); `oed/opt.py:59-60` reads `.information_gain` (absent from `ASRPredictedVariables`); `oed/gp.py:210-218` reads `.population_size`/`.diversification_ratio`.

Two asr scripts already import **proteinsmc's** `OEDDesign` alongside asr's `ASRDesign`: `asr/scripts/analyze_oed.py:11-12` and `asr/scripts/diagnose_gp.py:4-5`. The plan lists neither.

**Corrected entry:** `oed/structs.py` → `asr/oed/smc_design.py` (separate module). Deduplicate `OEDFeatureMode` only. Add `analyze_oed.py` / `diagnose_gp.py` to the phase-3 repoint list.

---

### B.2.4 `oed/opt.py` → asr "real payload, asr lacks it" *(high)*

**Plan claims:** finite-difference FIM determinant; a real payload asr lacks.

**Actually true.** It is not a Fisher Information Matrix. `oed/opt.py:59-63` takes forward differences of a **single scalar** (`base_prediction.information_gain`) w.r.t. six parameters and sets `fim[i,j] = d_info_i * d_info_j` — the rank-1 outer product ∇g∇gᵀ. `opt.py:68 return jnp.linalg.det(fim)` is therefore identically 0 for every design, and `oed/run.py:153-157 best_idx = jnp.argmax(fim_values)` selects over floating-point noise. Verified numerically: `np.linalg.matrix_rank(np.outer(d,d))` → 1, `det` → 0.0.

Secondary defect: `perturb_design` (`opt.py:11-15`) adds ε=1e-4 to integer fields cast to `int` at `run.py:96-99`.

"asr lacks it" is true but for the wrong reason — asr has a working alternative: masked uncertainty sampling, `asr/src/asr/oed/gp.py:276-313 score_candidates_batch`, jitted and called at `scripts/oed_exploration.py:181`.

**Corrected entry:** `oed/opt.py` → **DO NOT MOVE.** Fix or delete. Moving it ports a silent no-op into a repo whose acquisition works. Note the entanglement: `train_surrogate_model` lives in `run.py:108-137` and is imported by `tests/oed/test_opt.py:4`, so deleting `opt.py` and `run.py` together orphans that test.

---

### B.2.5 `utils/metrics.py` SPLIT: "SMC-intrinsic half (…, ESS)" / "analysis half → asr as its only consumer" *(high)*

**Plan claims:** `safe_weighted_mean`, `calculate_logZ_increment`, ESS stay as `kernels/diagnostics.py`; entropy/divergence/barrier-crossing → asr.

**Actually true.** Three failures:

- **ESS does not exist in `metrics.py`.** It is computed inline as `1.0/jnp.sum(weights**2)` at `sampling/particle_systems/smc.py:151` and `parallel_replica.py:311`.
- **The "SMC-intrinsic half" is dead.** `rg 'safe_weighted_mean|calculate_logZ_increment'` over all eight repos → definitions (`metrics.py:25,65`), the `utils/__init__.py:41,43,79,95` re-export, and `tests/utils/test_metrics.py`. Zero `src/` call sites; the real log-evidence comes from blackjax (`smc.py:149`).
- **asr consumes none of the analysis half.** The only consumer anywhere is `proteinsmc/oed/experiment.py:155,160,161,166,169` — circular. asr's own entropy is `asr/src/asr/metrics.py:10 sequence_entropy`; its one `shannon_entropy` is `from scipy.stats import entropy as shannon_entropy` (`asr/scripts/visualize_antigenic_uncertainty.py:20`) — a third-party function under a shared name.

The split is three-way, not two: **5** functions follow `experiment.py` (jeffreys/JS/shannon/barrier-crossing/geometric-mean); **2** are private helpers of those 5 (`calculate_position_entropy` called only at `metrics.py:133`; `kl_divergence` only at `:188,205`) and should not appear in a move table; **3** are dead (`outcome_shannon_entropy` at `:233` has zero hits anywhere including tests, plus the two above).

**Corrected entry:** delete the 3 dead functions (an API-surface change — two are in `utils/__init__.py`'s `__all__`); move the 5 live ones as an internal detail of whatever happens to `oed/experiment.py`; do not name the 2 private helpers.

---

### B.2.6 `sampling/initialization_factory.py` DELETE — "a single dispatch on a `sampler_type` string" *(destructive)*

**Plan claims:** exactly the switch the KERNELS registry removes; absorbed into each `Kernel.init()`.

**Actually true.** The file is **393 lines**; the string chain is ~30. Underneath:

- `_initialize_blackjax_smc_state` (`:186-250`) is a nine-way dispatch over **blackjax SMC algorithm variants** — `BaseSMC|AnnealedSMC|ParallelReplicaSMC|FromMCMC → smc.base.init`, `AdaptiveTemperedSMC`, `PartialPosteriors` (with `num_datapoints`), `TemperedSMC`, `CustomSMC` via `custom_init_fn`, plus two `NotImplementedError` branches. A registry keyed on *sampler type* does not subsume a dispatch keyed on *blackjax algorithm* — orthogonal axes.
- `_initialize_prsmc_state` (`:283-393`) is the entire PRSMC island constructor: ceil-based tile/repeat/truncate to `population_size_per_island`, a runtime `diversify_initial_sequences` call, a vmapped per-island `BaseSMCState` build and fitness evaluation, and construction of the `beta`/`mean_fitness`/`max_fitness`/`ess`/`logZ_estimate` `additional_fields` the PRSMC loop reads.

Live at `runner.py:28,277`; 25 tests in `tests/sampling/test_initialization_factory.py`; `tests/test_runner.py:63` and `tests/test_runner_jax_inputs.py:19,72` patch it. `docs/TESTING_COVERAGE_ASSESSMENT.md:229` records 92% coverage.

**Corrected entry:** DELETE → **SPLIT/REWRITE**. The blackjax dispatch and the PRSMC constructor must be ported into the kernels. Deleting the module deletes the 25 tests that currently encode the known int8→float32 bug — that stops testing for it, and the defect reappears inside `HMCKernel.init()`/`NUTSKernel.init()`.

*(Note: the repo disagrees with itself on the failing-test count — `CLAUDE.md:95` and `AGENTS.md:144` say 4; `docs/FAILING_TESTS_ANALYSIS.md:264-269` and `docs/TESTING_COVERAGE_ASSESSMENT.md:382` name 6. See §B.5.)*

---

### B.2.7 `utils/memory.py` and `models/memory.py` DELETE → xtrax *(destructive ×2)*

**Plan claims:** both replaced by `xtrax.tiling.MemoryBudget` + `device_memory_budget` + `lowered_memory_estimate`.

**Actually true.** Both are live, and the substitution changes the objective.

**Live:** `runner.py:36 from proteinsmc.utils.memory import auto_tune_batch_size`, called at `runner.py:131` under `runner.py:112 if config.memory_config.auto_tuning_config.enable_auto_tuning:`, which defaults **True** (`models/memory.py:47`). Six symbols are public exports (`utils/__init__.py:32-38`, `__all__` at `:74,78,84,87,90,98`). 25 tests in `tests/utils/test_memory.py`; `tests/test_runner.py:117` patches it and `:172` asserts it was called. `models/memory.py` is re-exported at `models/__init__.py:8` (`__all__` `:30,39`), runtime-instantiated at `oed/experiment.py:55 AutoTuningConfig(enable_auto_tuning=False)`, and built in the shared fixture at `tests/conftest.py:17,90`.

**Wrong output kind:** `auto_tune_batch_size` (`utils/memory.py:170-232`) **selects** a batch size by wall-clock benchmarking the real fitness function (`time.time()` + `jax.block_until_ready`, argmin on `avg_time_per_batch` at `:219`), and `runner.py:139-144` feeds the scalar back into `get_fitness_function(batch_size=...)`. `BatchPlanner` never selects a batch size — `AxisDecision.batch_size` is documented "Final batch size used (from spec)" (`xtrax/src/xtrax/tiling/plan.py:96-106`); it takes the batch size as given and chooses a **strategy** (Vmap vs SafeMap) to fit a byte budget. All five `AutoTuningConfig` fields (`probe_batch_sizes`, `max_probe_iterations`, `memory_safety_factor`, `performance_tolerance`, `enable_auto_tuning`) have no counterpart in `MemoryBudget(bytes, estimate)` (`xtrax/src/xtrax/tiling/budget.py:34-56`).

> **Verifier vs challenger:** the verifier justified this with "xtrax deliberately does not benchmark / neither measures elapsed time," asserted from the one API the plan named without searching xtrax. **Challenger wins on the reason, verifier's verdict stands.** `xtrax/src/xtrax/loop/compile_time_clock.py:100-110` is `time.perf_counter()` + `jax.block_until_ready()` around a jitted call — the same idiom; `xtrax/src/xtrax/devtools/gates/performance.py` and `_performance_probes.py` likewise. The correct statement is "the API the plan named cannot do this," not "xtrax cannot."

**Corrected entries:** both rows DELETE → **REWRITE**, and they stand or fall together (`AutoTuningConfig` exists solely to parameterize the probe loop). `MemoryConfig.batch_size`/`device_memory_fraction` map onto `AxisSpec.default_batch_size` + `device_memory_budget(fraction=...)`; the benchmarking half must be deleted on its own merits as a recorded behavior change. §7 line 476's verification ("`xtrax plan` reproduces the batch sizes the deleted heuristics chose") is the right gate and **is not satisfiable by the API it names.**

Also note the plan contradicts itself: §6.2 line 356 says xtrax is "the only implementation with compiler-derived memory budgeting"; §7 line 406 says "Memory-budget estimation (4 implementations)". The allocator-`bytes_limit` read exists five times (`xtrax/tiling/estimators.py:57`; `aminx/src/aminx/tiling/planner.py:109`; `aminx/src/aminx/host/plan.py:279`; `prolix/src/prolix/resource_guard.py:24`; `proteinsmc/utils/memory.py:61`) and `Compiled.memory_analysis()` three times (`xtrax/tiling/estimators.py:87`; `aminx/src/aminx/profiling/hlo_tools.py:61`; `aminx/src/aminx/profiling/sampler_profile.py:272`). §7 is right; §6.2 is wrong.

---

### B.2.8 `models/types.py` DELETE — "self-described legacy redirect" *(destructive)*

**Plan claims:** an internal second redirect layer; delete.

**Actually true.** The characterization is accurate (39 lines, pure re-export of 15 names from `proteinsmc.types`, docstring says "now redirected"). The implied deadness is not. Module-level **runtime** importers: `models/translation.py:9`, `models/fitness.py:12`, `models/mutation.py:9`, `models/gibbs.py:11`, and `models/__init__.py:22`, which re-exports `EvoSequence`/`NucleotideSequence`/`ProteinSequence` into `proteinsmc.models.__all__` (`:31,40,49`). TYPE_CHECKING importers: `utils/jax_utils.py:16`, `utils/jax_uuid.py:13`, `utils/initiate.py:14`, `utils/fitness.py:25`, `utils/blackjax_utils.py:13`, `utils/mutation.py:13`, `models/sampler_base.py:28`, `scoring/mpnn.py:36`, `scoring/cai.py:20`. Plus `tests/models/test_types.py:10`, `tests/utils/test_fitness.py:11`, `tests/utils/test_fitness_advanced.py:11`.

**Corrected entry:** schedule as a **codemod** over 15+ import sites, not a phase-1 file removal. Do not list it alongside `padding.py`, which genuinely has zero importers.

---

### B.2.9 `utils/jax_utils.chunked_map` → `xtrax.transforms.safe_map` *(high)*

**Plan claims:** a swap, with one caveat (the divisibility `ValueError`).

**Actually true.** Four differences, three unmentioned, two of which break immediately.

1. **Inverted tuple convention.** `chunked_map` unpacks: `jax_utils.py:55-58 def func_to_map(x): if isinstance(x, tuple): return func(*x, **kwargs)`. `safe_map` passes the tuple as one arg: `xtrax/src/xtrax/transforms/map.py:29-39 jax.vmap(fn)(xs)` / `jax.lax.map(fn, xs)`, with `fn: Callable[[T], T]`. Breaking sites: `utils/mutation.py:217-221` passes `(mutation_keys, population)` to a two-arg `fn(k, seq)` (`:208-212`); `runner.py:126-131` builds `(tune_key, jnp.zeros(...))` for `memory.py:137,142`. Both → `TypeError: missing 1 required positional argument`.
2. **No `static_args`.** `rg 'static_args' xtrax/src` → zero. Both `smc_tree.py` sites use it (`:83-87 static_args={"landscape": landscape}`, `:144-148 static_args={"ls": landscape}`).
3. **Empty-pytree behavior.** `chunked_map` returns empty arrays (`jax_utils.py:46-48`); `safe_map` raises `IndexError` at `map.py:26 jax.tree.leaves(xs)[0]`. Pinned by `tests/utils/test_jax_utils.py:34-41,81-90`.
4. **Silent vmap escalation.** `map.py:29-30 if batch_size is None or n <= batch_size: return jax.vmap(fn)(xs)` — materializes the whole population, the exact blow-up the chunking prevents. `chunked_map` always goes through `lax.map` (`jax_utils.py:60`).

Existing tests pin the divergent semantics: `tests/utils/test_jax_utils.py:57-66` asserts unpacking, `:68-79` asserts `static_args`, `:24-31` asserts uneven chunks (n=13, bs=5).

**On the divisibility caveat (X1b) — confirmed and understated.** `generate_tree_data_smc` defaults are `pop_size: int = 100` (`smc_tree.py:178`) and `inference_batch_size: int = 64` (`:182`); the two `chunked_map` sites map over the **population** axis. 100 % 64 = 36 → `map.py:32-36` raises on the first default-configured call. The plan audits the wrong axis.

**Third breaking site the verifier missed.** `smc_tree.py:10 from trex.utils.memory import safe_map`, called at `:290-297` over `expanded_paths` (leading axis = `num_leaves`, `:275-276`) with the same `inference_batch_size`. trex's `safe_map` (`asr/vendor/trex/src/trex/utils/memory.py:15-28`) is a bare `lax.map` passthrough — **no divisibility check, no vmap path** — i.e. semantically `chunked_map`, not xtrax's. §6.2 line 362 explicitly folds it into the consolidation, so that site inherits a raise it does not have today, on an unconstrained tree-derived cardinality.

> **Verifier vs challenger:** verifier flagged line 290 as "a third cardinality"; challenger read trex's implementation and showed it is a third *breaking site*. **Challenger wins.**

**Corrected entry:** not a name substitution. Rewrite all three call sites into single-arg closures (the `smc_tree.py:290` lambda already has this shape — `inputs[0]`/`inputs[1]` — which is itself the clearest demonstration of the inverted convention), replace `static_args` with closure capture, and choose divisibility policy explicitly. Note `xtrax/src/xtrax/tiling/plan.py:131-132,514-523` ships the non-divisible case as a warned "deferred-failure contract" — so `xtrax plan` will emit a plan that then raises in `safe_map`.

---

### B.2.10 `utils/annealing.py`'s `ANNEALING_REGISTRY` absorbs prolix's ladders *(high)*

**Plan claims:** `prolix/pt/temperature.py::generate_temperature_ladder` registers under `@register_schedule`.

**Actually true.** Four blockers, one of them a units inversion.

- **Wrong axis.** Registry schedules are indexed by **time**: `get_annealing_function` wraps `func(current_step=, n_steps=, beta_min=, beta_max=, ...)` returning a scalar (`utils/annealing.py:38-57`). `generate_temperature_ladder(n_replicas, min_temp, max_temp, geometric=True)` (`prolix/src/prolix/pt/temperature.py:13-16`) is indexed by **replica** — the whole `(n_replicas,)` vector held simultaneously in parallel tempering (`prolix/src/prolix/pt/replica_exchange.py:284`).
- **Wrong arity.** Scalar `ScalarFloat` vs `Float[ArrayLike, "n_replicas"]`.
- **Inverted quantity.** proteinsmc anneals **beta**, ascending `beta_min → beta_max` (`annealing.py:101-103`). prolix builds **temperature in Kelvin**, ascending (`prolix/tests/pt/test_replica_exchange.py:12` passes `(4, 300, 400)`). Since β = 1/k_BT, a geometric ladder ascending in T is descending in β. Registered as-is it anneals the wrong direction **with no type error** — the exact failure class `~/.claude/rules/BATHOS.md` was written after.
- **Signature incompatible + unjitted.** `generate_temperature_ladder` accepts none of `current_step`/`n_steps`/`beta_min`/`beta_max`; it branches on `geometric` in Python and uses `n_replicas` as a `jnp.linspace` shape, so it cannot join the `@partial(jit, static_argnames=...)` pattern of the four existing entries (`:74,122,182,233`).

**Corrected entry:** remove from the consolidation list. proteinsmc has no `geometric` entry today, so this was an addition presented as a merge.

---

### B.2.11 §3.1's FATAL objection: "both scorers already close over arrays, therefore both fail `static_argnames`" *(medium)*

**Plan claims:** `scoring/esm.py:39-42` and `scoring/mpnn.py:104-112` close over arrays and would fail `jax.jit(static_argnames=...)` with a non-hashable-static error.

**Actually true.** A closure hashes by **identity**; `hash()` never inspects `__closure__`. Reproduced: a `@jit` closure over a live `jnp.ones((4,4))` passes as a static argname (`static closure OK: 64.0`), and so does an *undecorated* one — which also refutes the companion claim that it "works only because `utils/fitness.py:94` returns a `@jit`-decorated closure" (line 94 is blank; the decorator is `:95`).

What genuinely fails is the **proposed** shape: `class F(eqx.Module): w: jax.Array` → `TypeError: unhashable type: 'jaxlib._jax.ArrayImpl'`, and as a static arg → `ValueError: Non-hashable static arguments are not supported` — both reproduced verbatim as §3.1 quotes them.

Citation error: `scoring/mpnn.py:104-112` is the `ar_mask = 1 - np.eye(...)` block; the closure over `processed_inputs.*` is the `partial(...)` at **`:115-122`**.

> **Verifier vs challenger:** verifier supported the refutation with "these two scorers pass through the current static boundary today; that is why the code runs at all." **Challenger wins on that sentence** — `make_mpnn_score` bails at `mpnn.py:80` because `PRXTEINMPNN_AVAILABLE` (`:14`) is False (`prxteinmpnn` does not resolve), and `rg -ln 'make_esm_score|make_mpnn_score' tests/ scripts/` → one file, `tests/scoring/test_esm.py`, which exercises the scorer standalone. Neither scorer has a call site through the jit boundary. The mechanism refutation stands regardless.

**Corrected entry:** `eqx.filter_jit` remains a correct phase-2 precondition, but it is a precondition of the **new** contract, not a fix for a live compilation failure. Cite §3.4 as the evidence, not the current scorers.

---

### B.2.12 "batched form is wrong for 3 of 8 kernels" *(medium)*

**Actually true.** No count of 8 exists in the code: `runner.py:54-79 SAMPLER_REGISTRY` has **6** entries; `ls src/proteinsmc/sampling/` gives 7 modules (`ste.py` unregistered and fitness-free — its objective is cross-entropy against `target_logits`, `ste.py:69-72`).

Point-form kernels are **4**, not 3: `gibbs.py:74-76` (vmaps over `n_states` variants of one sequence) and `:138`; `mcmc.py:69-72` (`fitness[0]` scalar); `hmc.py:65,90-93` (`jax.grad` of a scalar); `nuts.py:71,94-97`.

And **every current kernel consumes the point form** — `smc.py:121 jax.vmap(fitness_fn, in_axes=(0,0,None))`, `parallel_replica.py:289-294` nests two vmaps around a point call. The batched form is consumed by **zero** kernels today, so making it the default changes 100% of the kernel surface, not 5/8 of it.

**Corrected entry:** "wrong for 4 of 6 registered kernels; batched is a new default, not a preserved one."

---

### B.2.13 `models/fitness.py`'s `needs_translation` is "the mechanism by which codons leak into the fitness core" *(medium)*

**Actually true.** `models/fitness.py:85-91` is `jnp.where(jnp.array([f.n_states != n_states for f in self.fitness_functions]), 1, 0)` — an integer comparison. No codon table, no alphabet, no nucleotide symbol; the file imports neither `utils/constants.py` nor `utils/translation.py`. Shared word, unrelated thing.

The actual leak is `translate_func: TranslateFuncSignature`, declared at `utils/fitness.py:44,71`, **called** at `:55 seq, _ = translate_func(sequence, key_i, _context)`, and bound at `runner.py:98-100` to `nucleotide_to_aa if config.n_states == NUCLEOTIDES_NUM_STATES else aa_to_nucleotide` (codon imports at `runner.py:34,38`). The plan states this correctly at §5 point 1 and contradicts itself in the §4 table.

**Corrected entry:** the row should name `TranslateFuncSignature`. `codon.through_translation(fitness)` must absorb that callable; deleting `needs_translation` removes no codon dependency.

---

### B.2.14 §2's five-line CI grep enforces the no-cycle rule *(high)*

**Plan claims:** `grep -rn "aminx\|prxteinmpnn\|proxide" src/proteinsmc/` empty ⇒ no cycle (§2 line 43, §7 phase 3 line 470).

**Actually true.** The grep cannot see the cycle the same document creates. §3.4 (lines 178-182) declares `[project.entry-points."proteinsmc.fitness"]` in aminx; §4's `utils/fitness.py` row specifies `importlib.metadata.entry_points("proteinsmc.fitness")` in `registry.py`. Resolving an entry point executes `import aminx.fitness.mpnn`, whose module-scope `class MPNNFitness(psmc.Fitness)` re-enters proteinsmc — **with no `aminx` token anywhere in proteinsmc's source.** If `registry.py` resolves eagerly at `proteinsmc/__init__` import time, aminx binds against a partially-initialized module. The repo already uses runtime `importlib.util.find_spec` gating (`scoring/mpnn.py:5,14`), so this is an established pattern here, not hypothetical.

**Corrected entry:** replace the grep with (a) lazy, on-demand entry-point resolution — never at `__init__` import time — and (b) a test that imports proteinsmc alone in a clean venv with aminx installed and asserts no `aminx` module in `sys.modules`.

---

### B.2.15 §5's quarantine wall is "the same mechanism aminx already runs… six lines of TOML" *(high)*

**Plan claims:** a ruff `flake8-tidy-imports` banned-api rule forbidding `proteinsmc.codon` imports from `api.py`, `drive.py`, `kernels/`, `compose.py`, `registry.py` gives "type-level proof that the core is alphabet-agnostic."

**Actually true.** It is the inverse of aminx's use. aminx bans *other subsystems'* modules globally with one exemption (`aminx/pyproject.toml:141-157` + `src/aminx/potts/designer.py = ["TID251"]`). proteinsmc would ban its **own subpackage** and need exemptions everywhere except five files. Reproduced against real ruff 0.16.3 in a package layout matching proteinsmc's:

- The ban is **global** — it flags `src/pkg/io.py` (outside the five-module list) and, after adding self-imports, flags `src/pkg/codon/__init__.py` **twice**: the quarantined package importing itself.
- `importlib.import_module("pkg.codon")` → no TID251. Silent bypass.
- `proteinsmc/pyproject.toml:44 exclude = [".venv","venv","build","dist","__pycache__","tests"]` — the wall has **zero** effect on the 15 codon-touching test files.

> **Verifier vs challenger:** the verifier also reported `from ..codon import X` (parent-relative) as a bypass, from a mis-set-up reproduction. **Challenger wins** — in a proper `src/`-rooted package ruff flags `src/pkg/sub/parent.py:1:1 TID251` on exactly that line. One fewer bypass than claimed; the conclusion is unaffected.

**Corrected entry:** banned-api cannot express "ban in these five modules." Scoping requires `per-file-ignores = ["TID251"]` over the **complement** — including `proteinsmc/codon/**` itself, `io.py`, `landscapes/`, `runner.py`, `scripts/` — extended for every new file. `importlib` bypasses it; tests are outside its scope. Describe it as a smoke alarm, not a proof.

---

### B.2.16 §4's "`import proteinsmc` pulls in jax + xtrax and nothing else" *(medium)*

**Actually true.** `xtrax/pyproject.toml:7` declares `["jax>=0.10.2,<0.11", "jaxlib>=0.10.2,<0.11", "equinox>=0.11.0", "optax>=0.2.3", "orbax-checkpoint>=0.6.0", "grain>=0.2.0", "numpy>=1.26", "pytest-asyncio>=0.23"]` — including a **test package in the runtime dependency list**, and a hard single-minor jax ceiling against proteinsmc's bare unpinned `jax` (`proteinsmc/pyproject.toml:10-25`).

**Corrected entry:** §8's Q5 must cost the pin as three things, not two: alpha cadence, the 3.13 floor, and a single-minor jax ceiling plus five transitive runtime dependencies inherited by everything downstream in the §2 DAG. *(The 3.13 floor itself is CONFIRMED: `xtrax/pyproject.toml:6 requires-python = ">=3.13"`, and `xtrax/src/xtrax/transforms/map.py:8 def safe_map[T](` / `transforms/scan.py:6 def safe_scan[Carry, X, Y](` are PEP-695 — a SyntaxError on 3.11, so the bump is forced by the very function §5 proposes to adopt.)*

---

### B.2.17 §6.6's "4 vocabularies, 3 permutations" census *(medium)*

**Actually true.** The census excludes asr, which vendors proteinsmc as a live workspace member (`asr/pyproject.toml:108`) and is the plan's destination for `oed/`. asr adds:

- `asr/src/asr/alphabet_reconcile.py:29-33` — four orders in one block: `CANONICAL_Q20 = "ACDEFGHIKLMNPQRSTVWY"`, `LG_ORDER = "ARNDCQEGHILKMFPSTWYV"`, `CANONICAL_Q21 = "ACDEFGHIKLMNPQRSTVWY-"` (**gap** at 20, not X), `POTTS_Q21 = "-ACDEFGHIKLMNPQRSTVWY"` (**gap first**).
- Five more permutations: `asr/src/asr/alphabet.py:16-18,22` (`MPNN_TO_CANONICAL`, `CANONICAL_TO_MPNN`, `POTTS_TO_CANONICAL`), `mtt_training_pipeline.py:35-38` (`CANONICAL_TO_ESM`), `alphabet_reconcile.py:36` (`perm_to_canonical`).

**And "Potts" is itself a shared word across the ecosystem.** aminx's `POTTS_ALPHABET` (`aminx/src/aminx/potts/model.py:40`) is the 21-char MPNN string with an identity map (`:47`) — correctly not a fifth vocabulary. asr's "Potts order" is gap-first, a genuinely distinct alphabet. asr contradicts itself about this in its own tree: `asr/src/asr/alphabet.py:20-22 POTTS_TO_CANONICAL = jnp.arange(21)` ("we will assume it matches… for now") vs `alphabet_reconcile.py:5`'s gap-first declaration — wrong by a 21-cycle if the reconcile module is right. asr's own header (`alphabet_reconcile.py:1-11`) states the stakes: "A silent permutation between these orders produces ARBITRARY recovery-Hamming numbers with no error signal."

Also: the AF↔MPNN permutation duplication §6.6 flags lives at `proxide/src/proxide/io/parsing/mappings.py:25-43` vs `aminx/src/aminx/utils/aa_convert.py:19-35` — **not** at `proxide/chem/residues.py` as §4 implies.

**Corrected entry:** the census is at minimum 6 vocabularies / 8 permutations once asr is included, and the vocabulary reconciliation is the load-bearing part of §6.6 — not a tidy-up.

---

## B.3 Claims needing qualification

### B.3.1 `scoring/mpnn.py` → aminx — "already a correct thin adapter, just on the wrong side"

**Condition:** true only after four fixes, one of which changes the returned numbers.

1. **Alphabet.** No conversion anywhere in proteinsmc (`rg 'af_to_mpnn|mpnn_to_af|MPNN_ALPHABET|aa_convert' proteinsmc/src proteinsmc/tests` → exit 1) while aminx applies `af_to_mpnn` at four host sites before every model call. proteinsmc index 1 = Arg; aminx index 1 = Cys. This must be item one. Corroborating mislabel in the repo's own tests: `tests/utils/test_translation.py:72` comments `[0, 1, PROTEINMPNN_X_INT]  # A, C, X` — under the AF ordering in force, index 1 is R.
2. **Model type.** `make_mpnn_score(mpnn_model_params: ModelParameters, ...)` (`mpnn.py:65`) passes a bare PyTree where `aminx/src/aminx/scoring/score.py:57 make_score_fn(model: ModelProtocol)` requires an eqx.Module with `features`/`encoder`/`decoder`/`w_out`/`w_s_embed`/`capabilities` (`aminx/src/aminx/types/protocols.py:124-146`). Masked by `# type: ignore[arg-type]` at `mpnn.py:111`; `score.py:77` (`isinstance(model, eqx.Module)`) and `:80` silently no-op on a PyTree.
3. **`sequential` == `same_random`.** The local shim `sequential_decode_order` (`mpnn.py:53-61`) returns `jnp.arange(N)`; `aminx/src/aminx/utils/decoding_order.py:87-109 single_decoding_order` returns the identical identity permutation. `mpnn.py:90-97` wires them to the two settings, so both produce the same order and neither is random.
4. **~25% duplicates aminx.** `create_protein_dataset` prep duplicates `aminx/src/aminx/host/prep.py:102`; the AR mask (`mpnn.py:100-105`) duplicates `aminx/src/aminx/utils/autoregression.py:generate_ar_mask` (used at `score.py:118`).

**Never executed by a test:** `rg -ln 'mpnn' proteinsmc/tests/` → only `tests/utils/test_fitness.py`, whose sole reference is `:148 assert "mpnn" in FITNESS_FUNCTIONS`.

**The version being moved is stale.** `asr/vendor/proteinsmc` is a separate checkout on branch `fix/asr-3713-lazy-mpnn-import` at `9f42237` ("defer proxide/prxteinmpnn imports to call time"), three months newer than canonical `15abde8`. `diff -rq --exclude=__pycache__` between the two trees returns **exactly one** differing file: `scoring/mpnn.py`. The fork defers the imports because `proxide.ops.dataset` → `ops/transforms` → `physics/features` → `physics/electrostatics:11 from jax_md import space` → dm-haiku, which breaks on jax≥0.11.

**Consequence for the collapse recommendation:** routing through `aminx.host.prep` re-imposes that hazard — `aminx/src/aminx/host/prep.py:21` imports `create_protein_dataset` at **module scope**, and `aminx/pyproject.toml:11` pins `jax>=0.4.35` with no upper bound. Either keep a deferred import or aminx inherits asr issue #3713.

> **Verifier vs challenger:** verifier verified against the canonical tree and recommended the collapse. **Challenger wins on both counts** (stale source, and the collapse re-imposes the bug the fork fixed).

**Corrected entry:** move **from the asr fork**, insert `af_to_mpnn`, retype to `ModelProtocol`, delete the shim, disambiguate or collapse the two decoding settings, and keep `create_protein_dataset` deferred. Not a relocation.

---

### B.3.2 `scoring/esm.py` + `utils/esm.py` → aminx

**Condition:** three things must land in the same phase.

- **Coupling is real and runtime** — `scoring/esm.py:39-43` calls `load_model(...)` then `eqx.filter_jit` at module-factory time; `:62` calls `remap_sequences`. It cannot separate from `utils/esm.py`.
- **Destination path is wrong.** `ls aminx/src/aminx/` → `model/` singular, 13 files. No `models/`. Plan lines 146-148, 302 say `aminx/models/esmc.py`.
- **Not a drop-in.** `utils/esm.py` ships its own `AbstractFromTorch` port shims — `Linear`:83, `LayerNorm`:100, `Sequential`:122, `SparseEmbedding`:147 — reimplementations of `eqx.nn` primitives aminx uses natively (`rg 'AbstractFromTorch' aminx/src` → 0; `aminx/src/aminx/ebm/trunk.py:147-724` is built from `eqx.nn.Linear`/`eqx.nn.LayerNorm`). `load_model` (`esm.py:288`) collides with `aminx/src/aminx/io/weights.py:151 load_model`. And aminx has no PLM at all: `rg -w 'rotary|RoPE|rope' aminx/src` → 0 hits.
- **In its favor, and unstated by the plan:** the weight-loading infrastructure genuinely matches. `esm.py:299-302` does `hf_hub_download(repo_id="maraxen/esmc_models")` → `eqx.tree_deserialise_leaves(f, skeleton)`; `aminx/src/aminx/io/weights.py:19,98,139-148` does `hf_hub_download(repo_id=HF_REPO_ID="maraxen/aminx")` → same idiom. Same HF org, same pattern.
- **Charter widening.** ESM-C is a sequence-only masked LM over a 33-token vocab (`esm.py:437-471`); every aminx model is structure-conditioned over 21 letters. That is a decision to state, not infer from an import count.

**Six asr sites break, none of them listed.** §7 line 470 enumerates "asr's 10 import sites" and names no ESM one. Real: `asr/src/asr/mtt_training_pipeline.py:24,25,30`, `asr/src/asr/analysis/biophysics.py:7`, `asr/scripts/vram_stress_test.py:6`, `asr/tests/spikes/test_esmc_proteinsmc_spike.py:6`. Runtime instantiations at `mtt_training_pipeline.py:305-307` and `biophysics.py:16`. Constants reuse independent of `remap_sequences` at `mtt_training_pipeline.py:35-36`. Plus `asr/scripts/env_manifest.py:46`.

**A 1.33 GB file depends on the literal relative path.** `utils/esm.py:289-296` writes `esm_models/{name}.eqx` cwd-relative; `ls -la asr/esm_models/` → `esmc_300m.eqx`, 1,331,991,147 bytes.

**Corrected entry:** `utils/esm.py` → `aminx/model/esmc.py`; `scoring/esm.py` → `aminx/fitness/esm.py`. Prerequisites: port or quarantine the `AbstractFromTorch` shims; namespace `load_model`; migrate the weights path onto aminx's HF-cache convention or preserve the relative path; fix `remap_sequences` per §B.2.1 **before** the move, or the defect crosses a repo boundary; repoint six asr sites in the same phase.

*(The "aminx has zero ESM" negative is **CONFIRMED** — see §B.4's note on how it was searched. Do not re-derive it from the plan's `grep -rln esm src/aminx/`; that command's own re-run failed silently because `rg -ril` parses `-r` as `--replace`.)*

---

### B.3.3 `oed/gp.py` → asr — "asr's fork is more advanced; reconcile, don't overwrite"

**CONFIRMED with one addition.** asr's fork (`asr/src/asr/oed/gp.py:3` "Adapted from proteinsmc/oed/gp.py") strictly adds ARD (`:33,39-48` vs proteinsmc's scalar `gp.py:52`), masked static shapes (`:31,169-170,57-61`), `log_marginal_likelihood` (`:77-105`), BFGS hyperparameter optimization (`:108-153`), jitted batch acquisition (`:276-313`), and `features_to_design` (`:228-253`). Decisively, proteinsmc's `features_to_predicted_variables` does `del var_dict` (`gp.py:257`) — it discards GP variance, making uncertainty sampling impossible; asr's returns all six variances (`:267-272`).

**Addition:** asr's fork **dropped** proteinsmc's `X_mean`/`X_std` input standardization (`proteinsmc/oed/gp.py:26-27,75,136-138`), relying on ARD length-scales plus a manual `log10` on mutation_rate (`asr gp.py:217`). Reconciliation must restore it or record that ARD subsumes it.

---

### B.3.4 `oed/run.py` → asr — "the BO outer-loop CLI"

**Condition:** it is a documented entry point with an open review item, and it holds a symbol another moving file's test imports.

- `proteinsmc/CLAUDE.md:22` — `uv run python src/proteinsmc/oed/run.py  # OED outer loop`, one of four documented commands.
- `proteinsmc/docs/OED_TRACKING.md:45-48`, `docs/OED_TRACKING_SUMMARY.md:28`.
- `asr/260106/docs/WORKPLAN.md:109` — "[ ] Review `proteinsmc/src/proteinsmc/oed/run.py`" (open); `:151` invokes it.
- `train_surrogate_model` (`run.py:108-137`) is the only wrapper assembling `fit_gp_model` + `design_to_features` + `predict_with_gp_models` + `features_to_predicted_variables`, and `proteinsmc/tests/oed/test_opt.py:4` imports it. So the §B.2.4 "delete opt.py" and any "delete run.py" are entangled through a test neither claim names.

asr's `scripts/oed_exploration.py` (388 LOC, vmapped candidates + jitted batch scoring at `:44-50,181`) is genuinely better as a loop, but it is a research script, not a package entry point.

**Corrected entry:** DEFER. Two hard dependency conflicts must be sequenced first: `run.py:21-26` imports four symbols from `oed/tracking.py`, which the same table DELETEs; `run.py:12` imports from `experiment.py`, whose destination is contested (§B.2.2). Neither the table nor §7 sequences these.

---

### B.3.5 `oed/phase.py` → asr — "real payload"

**Condition:** dead in proteinsmc, but the destination case is stronger than the verifier found and weaker than the plan implies.

Dead: `rg 'proteinsmc.oed'` shows the only reference is `tests/oed/test_phase.py:2`. No `src/` caller; `run.py`/`experiment.py` never import it. Exhaustive delete check across all eight repos → `oed/phase.py`, `tests/oed/test_phase.py`, and the two vendored mirrors.

> **Verifier vs challenger:** verifier searched `detect_phase_boundaries|phase_boundar` in asr, found nothing, and recommended DELETE. **Challenger wins** — it searched the *concept*. asr runs a pre-registered program for exactly this: `asr/scripts/colab/sp65_boundary_map.bth.toml`, "SP-6.5 — Falsifiable boundary/null regime map," prediction P5 "the crossover from tie to win traces a K-mu frontier with K >= 2 and mu_gen >= 5," grid K∈{0,1,2,4} × mu_gen∈{1,5,10,50,100}, with a postmortem sidecar (it has been run). asr has the program and **no detector** — `rg -i 'crossover|jnp.gradient|np.gradient|frontier' asr --glob '*.py' -g '!vendor/**' -g '!aminx/**'` → zero in source; SP-6.5 decides P5 by the hand-coded `p5_ok = p2_ok and p3_ok`.

`detect_phase_boundaries` (`proteinsmc/src/proteinsmc/oed/phase.py:7-33`, argsort + `jnp.gradient` + magnitude threshold) is the only implemented boundary detector in the ecosystem.

**Corrected entry:** move to asr with the SP-6.5 work, not as a payload of the OED block. Deadness alone does not license delete when the receiving repo has a live pre-registered program for the analysis.

---

### B.3.6 `oed/smc_tree.py` → asr — "unimportable today"

**Condition:** refuted as a runtime fact, confirmed as a manifest fact; conclusion holds on lineage.

Runtime: `cd asr && .venv/bin/python -c "import proteinsmc.oed.smc_tree"` → **OK**, resolving to `asr/vendor/proteinsmc/src/proteinsmc/oed/smc_tree.py` (uv workspace puts proteinsmc and trex on one `sys.path`). `diff -rq asr/vendor/proteinsmc/src/proteinsmc/oed proteinsmc/src/proteinsmc/oed` → no differences. Manifest: `grep -n trex proteinsmc/pyproject.toml` → zero, so the trex dep is **undeclared** — but proteinsmc is uninstallable standalone anyway (`prxteinmpnn was not found in the package registry`), so the trex gap is never the binding failure.

Lineage supports the conclusion strongly: `smc_tree.py:8-11` imports `trex.nk_model`, `trex.types.Adjacency`, `trex.utils.memory.safe_map`, `trex.utils.types.EvoSequence`; returns `nk_model.PhylogeneticTree` (`:330-334`); `nk_model.get_fitness` is called inside `scan_body` (`:63,83`) — runtime, not type; and its own comment at `:210-212` cites "asr tests/test_smc_generator_invariants.py."

**Consequence the plan misses:** this is the most-imported proteinsmc module in asr (5 of 9 `proteinsmc.oed` imports) and is **already** the load-bearing edge §7's phase-3 gate demands. Moving it removes the last runtime edge rather than creating one — the gate is satisfied before the port.

---

### B.3.7 `oed/nk.py` → asr, "must import `proteinsmc.landscapes.NKLandscape`"

**Condition:** the stated hazard is theoretical; the real one is representational; and the constraint names a module that does not exist.

- No pytree-identity instance is live — nothing in asr constructs `proteinsmc.models.nk_landscape.NKLandscape`.
- The live incompatibility is **layout**: proteinsmc's is `fitness_tables: Float[Array, "N q q ... q"]` (`models/nk_landscape.py:33-45`); trex's is a plain dict with `(n, n_states**(k+1))` **flattened** (`trex/nk_model.py:17-42`). asr's OED path uses trex's exclusively (`asr/oed/experiment.py:30,98,178`) and **mutates** it — `asr/oed/experiment.py:142 landscape["k"] = design.K`, legal only on a dict, raising on a frozen `PyTreeNode`. And `proteinsmc/oed/smc_tree.py:63` already speaks trex's layout.
- `ls proteinsmc/src/proteinsmc/` shows no `landscapes/` — `proteinsmc.landscapes.NKLandscape` is a phase-2 deliverable, so the phase-3 acceptance criterion is contingent on unfinished work it does not gate on.
- The whole module is a 27-line passthrough whose body is `return generate_nk_model(key, design.N, design.K, design.q)` (`:27`) — a **runtime call edge** into `proteinsmc/utils/nk_landscape.py:140`, a 239-line module the plan itself (line 507) calls "unused by the only live consumer." Post-move it would have zero asr consumers (its only caller is `oed/experiment.py:97`, which §B.2.2 says cannot merge).

**Corrected entry:** resolve §8's Q4 (proteinsmc's NK vs trex's) **before** moving. As written this is a one-line wrapper with no consumer, carrying a fresh runtime edge — delete-or-inline, not move.

---

### B.3.8 `utils/pmap_utils.py` DELETE → `xtrax.distributed.sharding`

**Condition:** the delete is sound; the destination is wrong.

`rg -rn 'jax\.pmap|pmap' xtrax/src/` → **zero hits**. `xtrax/src/xtrax/distributed/sharding.py` is `ShardingPolicy(eqx.Module)`:13 with `get_partition_spec`:25 and `apply_to_pytree`:40 (which `tree_map_with_path`s each leaf into a `PartitionSpec` — it returns a **spec tree**, never moves an array), plus `get_device_mesh`:83 and `get_hardware_mesh_profile`:115. `ls xtrax/src/xtrax/distributed/` → only `__init__.py`, `init.py`, `sharding.py`.

`distribute` (`proteinsmc/src/proteinsmc/utils/pmap_utils.py:19-57`) reshapes to `(num_devices, -1, ...)`, runs `@partial(jax.pmap, axis_name="devices")` (`:49-51`), and flattens back. No counterpart.

> **Verifier vs challenger:** the verifier's entire destination check was "verified to exist: `sharding.py` is present." **Challenger wins** — that is an `ls`, not a destination check.

Also: `distribute` **is** public API — `utils/__init__.py:52` imports it and `:85` lists it in `__all__`, so `from proteinsmc.utils import distribute` works today. Two-file edit, not one file. No tests lost (`ls tests/utils/` has no `test_pmap_utils.py`; `docs/TESTING_COVERAGE_ASSESSMENT.md:370`'s 35% is import-time execution).

**Corrected entry:** "drop the pmap scheme, keep the mesh one." `xtrax.distributed.sharding.get_device_mesh` replaces `BaseSamplerConfig._initialize_device_mesh` (`models/sampler_base.py:153-185`, a genuine hand-rolled duplicate including the same device-count check). Note the mesh is *also* dead weight: `sampler_base.py:151` instantiates it on every config build, `io.py:64-69` skips it during serialization because it cannot be pickled, and no sampler ever places an array on it.

---

### B.3.9 `padding.py` DELETE — "zero importers; xtrax owns `Bucket`/`select_bucket`/`bucketize`"

**Condition:** the delete stands on deadness alone. Do not justify it as "xtrax owns this."

**Deadness confirmed independently three times.** `rg 'proteinsmc.padding|from \.\.?padding|import padding'` across all eight repos → only prolix's own unrelated `padding.py`. `find ~/projects -name 'test_padding*'` → no proteinsmc hit. Not in `src/proteinsmc/__init__.py`. Notebook and shell sweeps clean.

**But xtrax does not own it.** `xtrax/src/xtrax/tiling/bucket.py:1-19` states its design commitment explicitly: padding runs "on the host, **before** the JIT boundary… Padding host-side with NumPy" — and the body imports `numpy as np`. proteinsmc's `padding.py:17` imports `jax.numpy as jnp`; every function is device-side and traceable inside jit. Four concrete mismatches:

| | proteinsmc | xtrax |
|---|---|---|
| Axis | `pad_sequence` pads **trailing** (`padding.py:80-81`) | `bucketize` pads **leading** only (`bucket.py:105`) |
| Placement | device-side `jnp` | host-side `np`, pre-JIT |
| Overflow | `pad_population` **silently truncates** (`:60-62 return sequences[:target_population]`) | `bucketize` **raises** (`:99-103`) |
| Coverage | `create_particle_mask`, `create_sequence_mask`, `masked_mean`, `masked_sum` | **no counterpart** — `rg 'masked_mean|masked_sum' xtrax/src` → 0 |

Only `get_seq_bucket` ↔ `select_bucket` is a true match.

> **Verifier vs challenger:** the verifier's evidence claimed "every hit is inside `padding.py` itself." **Challenger wins on the search, verifier's verdict survives.** Running `rg 'create_sequence_mask|masked_mean|masked_sum' proteinsmc asr` returns six hits outside it: `asr/src/asr/oed/padding.py:97,183,196` (defs), `asr/src/asr/oed/experiment.py:23,62,136` (import + two live calls), `asr/aminx/src/aminx/tiling/buckets.py:103`. These are the same lineage, not name-sharing — all three module docstrings are near-identical ("Padding and masking utilities for static-shape JAX compilation"), all carry the identical `for bucket in BUCKETS: if x <= bucket: return bucket` / `raise ValueError(f"... exceeds all buckets {BUCKETS}")`, and proteinsmc `padding.py:117 masked_mean` and aminx `buckets.py:103` are byte-identical. The verifier's regexes (`from \.\.?padding`, `import padding`) structurally cannot match `from asr.oed.padding import (`. Nothing imports **proteinsmc's** copy, so the delete holds — but the "5th bucket-ladder copy, xtrax owns it" rationale collapses: asr's fork is live with entirely different ladders (`N_BUCKETS=(32,64,100,200)`, `K_BUCKETS=(2,4)`, `Q_BUCKETS=(4,20)`, `asr/src/asr/oed/padding.py:19-22`), and deleting proteinsmc's copy resolves no ecosystem duplication.

---

### B.3.10 `oed/tracking.py` DELETE → bathos

**Condition:** split the row. Only one third has a destination.

- **Provenance** (`save_oed_record`:45, `load_oed_manifest`:84, `create_oed_summary`, `add_oed_to_metadata`) → bathos: defensible. `bathos/src/bathos/git.py:18-40 capture_git_state` returns hash+branch+dirty, a strict superset of `proteinsmc/io.py:42`'s bare `rev-parse HEAD`.
- **Optimizer resume state** (`save_oed_checkpoint`:134, `load_oed_checkpoint`:177) — serializes the BO `design_history` for warm restart. `rg -i 'checkpoint|resume' bathos/src/bathos --type py` → **3 hits, all prose**: `prereg.py:446` (comment), `linter.py:702` (docstring), `cli.py:1821` ("resume polling" message). bathos has no checkpoint system.
- **ArrayRecord index bookkeeping** (`get_next_record_index`:108, `get_run_records_range`:305, `get_shared_arrayrecord_path`:329) — `record_start_idx`/`record_count` offsets so one run's slice can be read from a **shared** ArrayRecord. `rg -ln 'arrayrecord|ArrayRecord' bathos` → **zero files**. `oed/experiment.py:14,23-26` imports exactly these and feeds them to `read_lineage_data_range`.

Granularity mismatch: a 50-design BO loop inside one process is not 50 bathos runs, so `write_run` is the wrong shape for `save_oed_record` too.

> **Verifier vs challenger:** the verifier refuted bathos and then routed the remainder to asr with no search. **Challenger wins.** `ls asr/src/asr/oed/` → `__init__.py experiment.py gp.py padding.py structs.py` — **no tracking.py.** asr forked four files and deliberately left the fifth. `grep -rn 'import bathos|from bathos' asr/src asr/scripts` → zero. asr uses neither.

**Corrected entry:** git shell-out → bathos. The checkpoint and ArrayRecord halves have **no destination** — they stay with whichever repo keeps proteinsmc's own OED loop, a question §8 has not settled. Also orphans: 7 tests (`tests/oed/test_tracking.py`), `scripts/view_oed_tracking.py`, and two committed docs.

---

### B.3.11 `utils/serialization.py` DELETE

**Condition:** conditional, and the table row does not say so.

Rationale is correct: `create_sampler_output_skeleton(config)` (`:48`) derives population size (`:12`) and sequence length (`:31`) from the config purely so `eqx.tree_deserialise_leaves` has a correctly-shaped target (`io.py:162,197`). Live consumers: `oed/experiment.py:29,123` and `tests/test_end_to_end_io.py:14,59`. No asr consumer.

The plan already concedes the ordering at lines 186 and 535/R4, but the §4 row at line 315 reads unconditional and line 446 batches it with genuinely dead files.

**Corrected entry:** "DELETE **only after** §3.5's msgpack switch + JSON header land." Until then it is the only reader for every existing `.arrayrecord` on disk — deleting early makes archived runs unreadable with no migration path.

---

### B.3.12 `utils/key_management.py` / `utils/config_unpacker.py` DELETE

**Condition:** unused confirmed; not free.

Both are genuinely unimported (`rg` across all eight repos → only definitions + their own tests + `docs/TESTING_COVERAGE_ASSESSMENT.md:364-365` recording 100% coverage). Neither is re-exported from `utils/__init__.py`.

The plan's stated reason for rejecting adoption of `key_management` checks out against the implementation, not just the name: `split_key_for_sampler(key, n)` returns `tuple(jax.random.split(key, n+1))`, so adopting it changes the number of children drawn from the same parent and every downstream key.

Note the reverse-direction finding on `config_unpacker.py`: `:12 from proteinsmc.models import BaseSamplerConfig` is a **module-level runtime** import, so the file is a live import edge *into* `models/` despite nothing importing it. That strengthens the delete. Its own docstring at `:31` references `@with_sampler_config`, a decorator that does not exist anywhere — corroborating disuse.

**Corrected entry:** flag as tested-but-unused. Deleting removes `tests/utils/test_key_management.py` (7 tests) and `tests/utils/test_config_unpacker.py` (10 tests), both at 100% coverage. Coverage percentage moves — a reporting change to book, not a surprise.

---

### B.3.13 The three `__init__.py` deletions

**Condition:** the import question answers cleanly; the packaging risk the verifier raised **does not exist**.

Import surface: `oed/__init__.py` is a single docstring. `scoring/__init__.py` re-exports four factories; `sampling/particle_systems/__init__.py` re-exports two loops. Nothing imports the re-exported *names* — every consumer uses submodules (`utils/fitness.py:11`, `sampling/__init__.py:6-7`, `runner.py:29-32`). No import breaks.

> **Verifier vs challenger:** the verifier flagged this to the parent as one of three headline findings — that `[tool.setuptools.packages.find]` (`pyproject.toml:33-34`) is `find_packages()` and would drop the subpackages from the installed distribution, prescribing `[tool.setuptools.packages.find-namespace]`. **Challenger wins, decisively.** In pyproject.toml the `namespaces` option **defaults to True** (`setuptools/config/expand.py: def find_packages(*, namespaces=True, ...)`, dispatching to `PEP420PackageFinder`). Empirically, a minimal project with proteinsmc's exact stanza and a subdirectory lacking `__init__.py` discovers `['pkg', 'pkg.sub']` on setuptools 68/75/84. And `find-namespace` is not a key setuptools accepts — it appears nowhere in `_validate_pyproject/fastjsonschema_validations.py`; the real toggle is `namespaces = false` *under* `find`. Adding the prescribed remedy would hard-fail the build.

**Corrected entry:** all three are safe to delete on import grounds. **Do not add the `find-namespace` prerequisite.** One real sequencing note survives: `scoring/__init__.py`'s eager `from .mpnn import make_mpnn_score` is what triggers `scoring/mpnn.py:14`'s `find_spec("prxteinmpnn")` at package-import time — sequence its removal with the edge-flip work that deletes that gate.

---

### B.3.14 `_run_smc_chunk` / `_run_prsmc_chunk` → `eqx.filter_jit` is "mechanical"

**Condition:** mechanical for every call site that exists today; the hazard is new work.

> **Verifier vs challenger:** the verifier asserted a live break — `config_to_jax` (`models/sampler_base.py:336-348`) turns every `int` field into a jnp array, and `_run_prsmc_chunk` declares four of them static (`parallel_replica.py:257-260`) and uses `n_exchange_attempts` as a **shape** (`:95-100,349-355`). **Challenger wins on the call path.** `runner.py:274` is `_get_inputs(config)` and `:284-285` are `initialize_sampler_state` kwargs — a different function. `grep -n 'run_fn' runner.py` → the only invocation is line 315, inside `if config.sampler_type == "smc":` at `:304`. `grep -rn 'run_prsmc_loop(' --include=*.py .` → exactly two sites, both tests (`tests/sampling/particle_systems/test_parallel_replica.py:186,244`), both passing **Python ints** (`:165-168`, `:216-219`). Python ints are hashable, so `jax.jit` works today, and under `filter_jit` they fail `eqx.is_array` and stay static — `jnp.zeros(n_exchange_attempts)` still works.

**Corrected entry:** the swap is mechanical for the call sites that exist. **But** if the plan also wires a prsmc dispatch branch through `config_to_jax` — which §B.5 item 1 requires — those four params arrive as jnp arrays and must be explicitly re-staticised (Python ints or `eqx.field(static=True)`), or `jnp.zeros(<traced>)` raises `ConcretizationTypeError`. Book it as new-work risk, not a latent bug.

**Separate, unconditional:** adopting the batched `Scores` form deletes `jax.random.split(key_for_fitness_fn, state.sequence.shape[0])` (`smc.py:116`) and the per-particle `jax.vmap(fitness_fn, in_axes=(0,0,None))` (`smc.py:121`), **changing the PRNG stream** — the exact objection §4 uses to reject adopting `key_management.py`. Apply the same criterion; golden values are invalidated.

---

### B.3.15 The codon-path enumeration ("13 files") and the `sequence_type` threading

**Condition:** the count is wrong in both directions and the severity ranking is off.

`rg -l -i 'codon|nucleotide|sequence_type|translat' src/ -g '*.py'` → **19 files**; the parenthetical enumerates **11** while claiming 13. Omitted and real: `models/fitness.py:16,85-91`; `utils/fitness.py:24,44,55,71,83,93` (the `translate_func` threading — the actual leak); `models/parallel_replica.py:29`; `models/types.py:9,11,20,27,29,38`; `models/__init__.py:15,24,42,47,49`; `utils/__init__.py:53,58-71,76-77,94` (re-exports every codon constant plus `aa_to_nucleotide`/`nucleotide_to_aa`). Test side: **15** files including `tests/conftest.py`.

> **Verifier vs challenger:** the verifier headlined `models/smc.py:26,28`'s `PopulationSequences` as "the most consequential omission… inside the very core the quarantine protects." **Challenger wins.** `sampling/particle_systems/smc.py:17` opens `if TYPE_CHECKING:`; the import at `:23` is inside it, under `from __future__ import annotations`, and its sole use at `:118` is a parameter annotation — erased at runtime. The definition (`models/smc.py:25-27`) is a jaxtyping alias whose "nucleotide" content is a *dimension-name string*; nothing in the file imports `utils/constants.py` or `utils/translation.py`. Reclassify as naming hygiene, ranked below `translate_func`.

Mitigating and worth recording: `sequence_type` is a `str`, and `config_to_jax` skips strings (`models/sampler_base.py:337-339`), so it threads host-side only (`runner.py:210,234,279,293`; `initialization_factory.py:36,98-99,138`) and never crosses the jit boundary.

---

### B.3.16 "The codon path collides with nothing in the ecosystem"

**Condition:** true where load-bearing; false as phrased; and asr is missing from the search.

**Load-bearing half CONFIRMED.** No genetic code table, codon→AA map, translation routine, CAI implementation, or nucleotide substitution model in proxide, aminx, prolix, xtrax, bathos or trex. Searched: `rg -i 'codon|nucleotide' <repo>/src --glob '*.py' --glob '*.rs'` → 0 in proxide, aminx, prolix, xtrax and bathos. **Corrected by Addendum C:** this originally said "0 in all," which included trex — trex actually returns **5 hits across 2 files** (`trex/ground_truth.py`, `trex/evals/benchmark.py`), one of which this very section reports three paragraphs below as an unrecorded collision. The conclusion (no genetic code table anywhere outside proteinsmc) survives and was independently re-verified; the stated evidence for it was self-contradictory. Targeted `rg '"ATG"|"TAA"|"TAG"|"TGA"|"TTT"'` and `rg 'JC69|HKY|GTR|K80|F81|TN93|substitution_model'` → the only triplet hits are GLYCAM carbohydrate residues coincidentally named TAA/TGA in force-field XML, and the only phylo-model hits are in `asr/src` (`felsenstein.py:87,90`, `lg_model.py:143`), **none in trex**. trex's `sankoff.py:30,104-132` takes an abstract `cost_matrix` + `n_states`, never a DNA rate matrix.

**False as phrased.** proxide deliberately syncs and ships AMBER nucleic force fields (`proxide/scripts/sync_forcefields.py:52-57` → `assets/amber/DNA.bsc1.xml`, `RNA.OL3.xml`); `prolix/viewer/src/types.ts:9` declares `atomType?: "P" | "D" | "R" | "L"`. Both atom-indexed, so the plan's own qualifier holds — narrow the sentence to "zero codon/translation handling."

**asr was excluded and has content.** `rg -c -i 'codon|nucleotide' asr --glob '*.py'` → **14 files**, including production source: `asr/src/asr/viral_utils.py:47,60` (`db="nucleotide"` NCBI fetchers), `asr/src/asr/nk_utils.py:255,413`, `asr/src/asr/site_independent_map.py:55`, `asr/src/asr/utils/gap_density.py:19`, plus `asr/scripts/debug/260520_iqtree_state_schema.py:24 ALPHABET = ["A","C","G","T"]` and `asr/scripts/experiments/prepare_nk_data.py:70`. None is a codon table — all are interop/preprocessing — so the **placement** decision survives and is strengthened (the quarantined subpackage has a plausible consumer). But asr also holds a second physical copy of the modules being quarantined (`asr/vendor/proteinsmc/src/proteinsmc/scoring/{cai,esm,mpnn,nk,combine}.py`), so the move carries a sync obligation the plan does not name.

**Unrecorded collision:** `trex/ground_truth.py:20-52 mutate()` — docstring "Apply random mutations to a population of nucleotide sequences… n_states: e.g. nucleotide types (4 for A,C,G,T)" — implements `offsets = randint(minval=1, maxval=n_states); (sequence + offsets) % n_states`, the **same** offset-mod-q kernel as `proteinsmc/utils/mutation.py:22-44 mutate`, which §4 routes to `proposals.PointMutation(rate, q)`. Near-verbatim shared docstring: a fork, not a coincidence.

---

### B.3.17 §1's "aminx structurally cannot host an iterative sampler"

**Condition:** the premise is false; the conclusion may still be right for other reasons.

See §B.1. The wall bans `aminx.inference.decode`/`host.plan`/`types.stages`/`inference.logits` — a pipeline, not a pattern — and `aminx/src/aminx/potts/sampling.py` (618 LOC) already hosts single-site Gibbs + parallel tempering under it. The direct collisions: `_parallel_tempering_exchange`:289 / `_attempt_adjacent_swap`:254 vs `proteinsmc/sampling/particle_systems/parallel_replica.py:50-58 migrate`; `_conditional_logits_site`:63 + `gibbs_sweep`:195 vs `proteinsmc/sampling/gibbs.py:73-79`.

**Corrected entry:** §1 must argue "proteinsmc keeps the sampler contract" on grounds other than aminx's inability, and §7's collision matrix must record the two collisions line 9 denies. This does not by itself change the destination — proteinsmc may still be the right home — but the argument as written is the failure mode this audit exists to catch, appearing in the plan's own foundation.

---

### B.3.18 Smaller qualifications, recorded for completeness

| Claim | Qualification |
|---|---|
| §7 "asr's 10 import sites" | **15** import statements across 13 files: 9 in the oed block (7 files), 6 in the ESM block (4 files). `smc_tree.py:13-14` and `oed/nk.py:5` are internal imports of *moving* files — counting them double-counts. The plan's own arithmetic (8+2+1) gives 11, presented as 10. Non-import references also affected: `asr/scripts/experiments/sp1_4_plan_b_driver.py:398`, `asr/pyproject.toml:16,76,108`, `asr/scripts/bootstrap_vendor_proteinsmc.sh`, three Colab notebooks, `asr/tests/test_generator_adapters.py:155,163`. **The gate rationale itself is CONFIRMED**: all 15 hit either `oed/` or ESM, so after phase 3 asr imports zero proteinsmc. |
| `convert_design_to_config` "always builds an NK evaluator; no protein content" | Attributed to the wrong function. `experiment.py:37` takes `fitness_evaluator` as a **required parameter** and passes it through at `:70`. It also branches on protein vs nucleotide at `:67-69`, and the integration test is named `test_run_oed_experiment_small_nk_protein` (`tests/oed/test_experiment_integration.py:14`). The NK hard-wiring is five lines in its only caller, `run_oed_experiment` (`experiment.py:100-104`). Do not use this as the load-bearing argument that `oed/` is a research harness. |
| `runner.py:303` "dispatches only smc" | **CONFIRMED, more severe than stated.** The branch is at `:304`, `final_state = None` at `:303`. `SAMPLER_REGISTRY` (`:54-79`) holds six live entries and **is** consulted for validation at `:176-191` — so the other five validate, resolve, leave `final_state = None`, and fall through to the write path having executed nothing. A silent no-op, not an error. Relatedly, `nuts` is currently **registered** (`:75-78`), so §4's "keep it unregistered" is a de-registration. |
| Fitness-signature enumeration ("three in `src/` plus a test mock") | At least **five**. The fifth is `sampling/initialization_factory.py:350-368,376`: its comment says "It expects a BATCH of sequences," it calls `fitness_fn(k, island_pop, None)` on a whole `(pop_size, seq_len)` island with no per-particle vmap, and indexes `initial_fitness_batch[:, :, 0]`. But `utils/fitness.py:116` returns shape `(1+n_fns,)` per sequence and `smc.py:121` vmaps per particle. Same factory output, two incompatible consumption shapes inside `src/`. |
| `ste.py` "takes no PRNG key, no population axis, no fitness, no state" | Three of four correct (`rg 'key\|PRNG' sampling/ste.py` → 0; objective is `get_loss`:48-72; returns `Logits` only). **Population axis present**: `ste.py:94 return jax.vmap(run_optimization_loop, in_axes=(0, 0, 0))`, while the declared `STELoopFn` (`:22-24`) is unbatched — the annotation contradicts the return. That is a *stronger* REWRITE argument than the one given: caller-side vmap over an undeclared axis, the exact anti-pattern §3.2 opposes. (The `jax.example_libraries.optimizers` finding is CONFIRMED and load-bearing: `:18` imported, `:81,84,87,89-92` used; no optax anywhere in the file.) |
| `xtrax.tiling` is a general kernel-op layer | Genuine at the AxisSpec/BatchPlanner level, with one real cross-domain consumer (`aminx/src/aminx/tiling/axes.py:24` + twelve constructions at `:27-131`) — **not two.** The verifier cited `prolix/src/prolix/tiling/axes.py:5,15,25` as the second; those construct **prolix's own** `AxisSpec` (`prolix/src/prolix/tiling/planner.py:56-64`, which has `axis_index` and `doc`; xtrax's `plan.py:31-53` has `dedup_eligible`/`bucket_boundaries`/`role` instead). prolix reaches xtrax only through `prolix/src/prolix/tiling/xtrax_adapter.py:33-52`, whose docstring documents an incompatible sentinel convention needing a three-branch fixup. So §6.2's proposed "~150-line `_xtrax.py` adapter modeled on prolix's" would be the **third** adapter, leaving three parallel AxisSpec dialects. Real limits: leading-axis only, statically-known cardinality (`plan.py:47 cardinality: int`), and `budget`/`memory_estimator` are mutually exclusive (`plan.py:178-182`). |

---

## B.4 DELETE claims — verification status

| File / symbol | Plan's claim | Verdict | What was searched | Safe to delete? |
|---|---|---|---|---|
| `padding.py` | Dead; 5th bucket ladder, xtrax owns it | **CONFIRMED (deadness); rationale REFUTED** | `rg 'proteinsmc.padding\|from \.\.?padding\|import padding'` + symbol sweep (`SEQ_BUCKETS\|MAX_POPULATION\|get_seq_bucket\|pad_population\|create_particle_mask\|create_sequence_mask\|masked_mean\|masked_sum`) across all 8 repos; `find ~/projects -name 'test_padding*'`; `-g '*.ipynb'` and `-g '*.sh'` sweeps; `rg 'masked_mean\|masked_sum' xtrax/src` → 0 | **YES** — zero importers, no test file. Delete on deadness alone. Strike "xtrax owns this": `bucket.py:1-19` is host-side NumPy pre-JIT, leading-axis only, raises where proteinsmc truncates, and has no counterpart for the four masking helpers |
| `utils/jax_uuid.py` | Byte-for-byte copy of `jax_utils.py` | **CONFIRMED** | Function-text SHA256 of both files → `generate_jax_uuid` `87a9a4de9bca` in both, `generate_jax_hash` `633c962a928c` in both; `rg 'jax_uuid\|generate_jax_uuid\|generate_jax_hash'` all 8 repos; `docs/TESTING_COVERAGE_ASSESSMENT.md:18,371` (0% coverage, no tests) | **YES** — the only importing test (`tests/utils/test_jax_utils.py:6-8`) imports from `jax_utils`, not `jax_uuid`. Sequencing: the surviving `jax_utils.py:16` still TYPE_CHECKING-imports `UUIDArray` from `models/types.py`; repoint at `proteinsmc.types` when that codemod lands |
| `models/lineage.py` | One-line docstring stub | **CONFIRMED** | `wc -l` → 1; full read; `rg 'models\.lineage\|from \.lineage\|import lineage'` all 8 repos → 0; `models/__init__.py` read in full; `ls tests/models/` → no test | **YES** — free. Do not let `io.py:144-172`'s `read_lineage_data`/`read_lineage_data_range` confuse the codemod; unrelated, own tests in `tests/io/test_io.py` |
| `models/protocols.py` | Abandoned second contract, TYPE_CHECKING-only reference | **CONFIRMED** | `rg 'models\.protocols'` all 8 repos → **one** hit, `sampling/particle_systems/smc.py:23`, inside the `if TYPE_CHECKING:` opened at `:17` under `from __future__ import annotations`; `docs/TESTING_COVERAGE_ASSESSMENT.md:15,85` (0%, 8 statements missed); `ls tests/models/` → no `test_protocols.py` | **YES**, with a repointing caution: `models/fitness.py:17`'s `FitnessFn` is a **different** contract (third arg `PyTree\|Array\|None` vs `protocols.py:16`'s `dict[str, Array\|float\|int]\|None`). A find-and-replace to the same-named alias silently changes what `smc.py` claims about its callers |
| `utils/key_management.py` | Tested, imported by nothing | **CONFIRMED (unused)** | `rg 'key_management\|split_key_for_sampler\|split_key_batched'` all 8 repos → only the definitions, `tests/utils/test_key_management.py` (7 tests), `docs/TESTING_COVERAGE_ASSESSMENT.md:365`; absent from `utils/__init__.py` | **YES** — but not free: deletes 7 tests at 100% coverage. Record the coverage delta |
| `utils/config_unpacker.py` | Tested, unused, hostile to a legible contract | **CONFIRMED (unused)** | `rg 'config_unpacker\|with_config\|with_sampler_config'` all 8 repos → definition + `tests/utils/test_config_unpacker.py` (10 tests) + coverage doc; aminx/proxide hits are unrelated (`aminx/tests/inference/test_vmap_axis_contract.py:419` local `process_with_config`; `proxide/crates/proxide-core/src/processing/residues.rs:51,85,88` Rust) | **YES** — deletes 10 tests. Note `:12`'s module-level runtime import into `models/` disappears with it, which is a small win |
| `utils/pmap_utils.py` | Dead → `xtrax.distributed.sharding` | **QUALIFIED** | `rg 'pmap_utils\|\bdistribute\('` all 8 repos → def + `utils/__init__.py:52,85` + coverage doc only; `rg -rn 'jax\.pmap\|pmap' xtrax/src/` → **0**; `ls xtrax/src/xtrax/distributed/` | **YES to delete, NO to the stated destination.** Two-file edit: also remove the `__all__` entry at `utils/__init__.py:85` (public API today). Retitle the destination — `get_device_mesh` replaces `sampler_base.py:153-185`, not `distribute` |
| `utils/memory.py` | Dead, replaced by xtrax | **REFUTED** | `runner.py:36,112,131`; `models/memory.py:47` default True; `utils/__init__.py:32-38` + `__all__`; `tests/utils/test_memory.py` (25 tests); `tests/test_runner.py:117,172` | **NO. Do not delete until** the batch-size selection is either reimplemented on top of `MemoryBudget`+`device_memory_budget` or explicitly abandoned as a recorded behavior change, and §7 line 476's gate is rewritten to something `MemoryBudget` can satisfy |
| `models/memory.py` | Dead → `xtrax.tiling.MemoryBudget` | **REFUTED** | `models/__init__.py:8,30,39`; `oed/experiment.py:15,55` runtime instantiation; `sampler_base.py:21,60,126-127`; `runner.py:112,120,131`; `tests/conftest.py:17,90`; `tests/models/test_memory.py:6,24,35` | **NO. Do not delete until** `utils/memory.py` is resolved — they are one decision. `tests/conftest.py` is the shared fixture; deleting breaks fixtures far beyond `test_memory.py` |
| `models/types.py` | Legacy redirect layer | **REFUTED (as a delete)** | `rg 'models\.types\|models import types'` all 8 repos → 15+ files: 5 runtime, 9 TYPE_CHECKING, 3 tests; `models/__init__.py:22,31,40,49` | **NO as a file removal. Schedule as a codemod**: rewrite 15+ import sites, delete `tests/models/test_types.py`, repoint `models/__init__.py:22`'s re-export at `proteinsmc.types` |
| `utils/serialization.py` | Superseded by `SamplerKernel.skeleton()` + msgpack | **QUALIFIED** | `rg 'create_sampler_output_skeleton\|utils\.serialization'` all 8 repos → `oed/experiment.py:29,123`, `tests/test_end_to_end_io.py:14,59`; no asr consumer | **NO YET. Do not delete until** §3.5's msgpack switch + JSON header land (plan already concedes at 186/535, table row does not). It is the only reader for existing `.arrayrecord` files |
| `sampling/initialization_factory.py` | A single string dispatch | **REFUTED** | `wc -l` → 393; read in full; `runner.py:28,277`; 25 tests; `tests/test_runner.py:63`, `tests/test_runner_jax_inputs.py:19,72` patch it; coverage 92% (`docs/TESTING_COVERAGE_ASSESSMENT.md:229`) | **NO. Retitle SPLIT/REWRITE.** Port `_initialize_blackjax_smc_state` (`:186-250`, nine-way blackjax dispatch) and `_initialize_prsmc_state` (`:283-393`, island constructor + `additional_fields`) into the kernels first. Fix the int8→float32 dtype bug on the way through — deleting the module deletes the tests that catch it |
| `oed/tracking.py` | → bathos | **REFUTED (2 of 3 parts)** | `rg -i 'checkpoint\|resume' bathos/src/bathos` → 3 prose hits (`prereg.py:446`, `linter.py:702`, `cli.py:1821`); `rg -l 'arrayrecord\|ArrayRecord' bathos` → **0 files**; `ls asr/src/asr/oed/` → no tracking.py; `grep -rn 'import bathos' asr/src asr/scripts` → 0 | **NO. Split.** Provenance quartet → bathos: OK. `save_oed_checkpoint`/`load_oed_checkpoint` (`:134-204`) and `get_next_record_index`/`get_run_records_range`/`get_shared_arrayrecord_path` (`:108,305,329`) have **no destination anywhere** — do not delete until §8 settles who owns proteinsmc's OED loop. Also orphans `tests/oed/test_tracking.py` (7 tests), `scripts/view_oed_tracking.py`, `docs/OED_TRACKING{,_SUMMARY}.md` |
| `oed/opt.py` | (implicitly kept — "real payload") | **REFUTED as a payload** | `opt.py:59-68` read in full; `np.linalg.matrix_rank(np.outer(d,d))` → 1, `det` → 0.0; `rg -in 'fisher\|fim\|perturb_design' asr -g '!vendor/**'` → only unrelated (`aminx/parity/evidence.py:154`, `model/packer.py:647`) | **Do not move. Fix or delete.** If deleted, `run.py:108-137 train_surrogate_model` must be preserved elsewhere — `tests/oed/test_opt.py:4` imports it |
| `oed/phase.py` | (move as payload) | **DEAD in proteinsmc; destination exists** | `rg -l --no-ignore 'detect_phase_boundaries'` all 8 repos → `oed/phase.py`, `tests/oed/test_phase.py`, two vendored mirrors; `rg -in 'crossover\|jnp.gradient\|np.gradient\|frontier' asr --glob '*.py' -g '!vendor/**' -g '!aminx/**'` → 0 in source | **Do not delete.** asr has a live pre-registered program (`asr/scripts/colab/sp65_boundary_map.bth.toml`, P5 K-μ frontier, already run) and no detector. Move with the SP-6.5 work |
| `scoring/__init__.py` | Flattened | **CONFIRMED (imports)** | `rg 'from proteinsmc\.scoring import'` all 8 repos → sole proteinsmc hit is `utils/fitness.py:11` (submodules); aminx hits are its own package | **YES.** Sequence with the edge-flip: the eager `from .mpnn import make_mpnn_score` is what triggers `mpnn.py:14`'s `find_spec` at package-import time. **Do NOT add the `find-namespace` prerequisite** — see §B.3.13 |
| `oed/__init__.py` | (DELETE, no justification given) | **CONFIRMED** | Read in full — a single docstring, no re-exports | **YES** — harmless beyond making `proteinsmc.oed` an implicit namespace package |
| `sampling/particle_systems/__init__.py` | (DELETE, no justification given) | **CONFIRMED (imports)** | `rg 'from proteinsmc\.sampling\.particle_systems import'` all 8 repos → no consumer of the re-exported names; `sampling/__init__.py:6-7` and `runner.py:29-32` use submodules | **YES**, same non-caveat as above |
| `io.py:create_metadata_file` git shell-out | → bathos | **CONFIRMED** | `bathos/src/bathos/git.py:18-40 capture_git_state` returns hash+branch+dirty with `_UNKNOWN` fallback at `:15`; `bathos/catalog.py:16-52`; `rg 'rev-parse'` across all 8 repos' `src/` → one library site outside bathos/xtrax (`proteinsmc/io.py:42`) | **YES.** Correct "5-way capture" to "one library-code site plus several `scripts/` uses" |
| `outcome_shannon_entropy`, `safe_weighted_mean`, `calculate_logZ_increment` | (implicitly "SMC-intrinsic, keep") | **DEAD — should be DELETE** | Per-function `rg -l --no-ignore` all 8 repos: `outcome_shannon_entropy` (`metrics.py:233`) → definition + vendored mirror only, no test; the other two → definitions + `utils/__init__.py:41,43,79,95` + `tests/utils/test_metrics.py` only | **YES to delete** — but note two of the three are in `utils/__init__.py`'s public `__all__`, so it is an API-surface change. This inverts the plan's row: the "SMC-intrinsic half" is the dead half |

---

## B.5 Unverifiable claims

**1. The `initialization_factory` failing-test count (4 vs 6), and the current pass/fail state of the entire suite.**

The repo contradicts itself: `CLAUDE.md:95` and `AGENTS.md:144` say 4 failing tests; `docs/FAILING_TESTS_ANALYSIS.md:264-269` names **six** (`test_initialize_parallel_replica_state`, `test_mcmc_initialization`, `test_smc_state_initialization`, `test_smc_state_with_none_beta`, `test_prsmc_state_initialization`, `test_prsmc_additional_fields_values`) and `docs/TESTING_COVERAGE_ASSESSMENT.md:382` agrees on 6.

Could not settle by execution. `uv run pytest tests/sampling/test_initialization_factory.py` aborts at dependency resolution: *"Because prxteinmpnn was not found in the package registry and your project depends on prxteinmpnn … requirements are unsatisfiable"* (`pyproject.toml:20` — the plan's Phase −1 cites line 21; it is line 20).

**Evidence that would settle it:** land Phase −1 (drop or rename the `prxteinmpnn` dep), then `uv sync --extra dev && uv run pytest tests/ -q --tb=no > /tmp/baseline.txt`. Until that exists there is **no baseline**, so no phase can claim "tests still pass." Every runtime check in this audit had to be run inside asr's venv against `asr/vendor/proteinsmc/src`.

**2. Whether `_run_prsmc_chunk` is reachable at all in the intended end state.**

`run_experiment` currently executes only the `"smc"` branch (`runner.py:304`), and the only `run_prsmc_loop` callers are two tests passing Python ints. Whether the `filter_jit` conversion is mechanical or hazardous depends entirely on how the plan wires prsmc dispatch — which it does not specify.

**Evidence that would settle it:** the §3.6 target call sequence for a non-smc sampler, written out, showing whether config ints reach `_run_prsmc_chunk` via `config_to_jax` (`models/sampler_base.py:336-348`) or as Python scalars.

**3. Whether landing the alphabet fix (§B.2.1) changes any asr result that has already been published.**

`asr/src/asr/analysis/biophysics.py:89` has been producing ESM and MPNN site scores under mismatched alphabets. Whether those numbers are in a preregistered claim, a figure, or a campaign is not determinable from source.

**Evidence that would settle it:** `mcp__bathos__find_runs` / `mcp__bathos__query_attestation` over asr's catalog for runs touching `compute_esmc_site_scores` or `get_fidelity_report`, plus `asr/scripts/colab/*.bth.postmortem.toml`. This is a research-integrity question, not a migration question, and it should be answered **before** the fix silently changes the numbers.

---

## B.6 Pre-phase-3 gate

Ordered. Each item is a command to run or a file to read, with a pass condition.

**G0 — Establish a baseline. Nothing else is checkable without it.**
```bash
cd ~/projects/proteinsmc
grep -n prxteinmpnn pyproject.toml                 # expect line 20
# after Phase -1 edit:
uv sync --extra dev && uv run pytest tests/ -q --tb=no | tail -5 > /tmp/baseline.txt
```
**Pass:** `uv sync` succeeds and a failure count is recorded. Reconcile it against `CLAUDE.md:95` (4) and `docs/FAILING_TESTS_ANALYSIS.md:264-269` (6); fix whichever document is wrong.

> ⚠ **Gate order revised by Addendum C.** The original list ran G1 (fix the alphabet) before G2 (determine which published asr numbers change) — which contradicts B.5 item 3's own instruction that the research-integrity question "should be answered **before** the fix silently changes the numbers." It also scoped G1's pass condition onto `scoring/mpnn.py`, a function that **cannot execute** (see C1), while omitting `asr/analysis/biophysics.py`, where the corruption is actually live. G1 as originally written could pass green while the bug shipped. The corrected order is **G0 → G2 → G-α → G1**, and G1's pass condition is rewritten below. Do not use the superseded ordering.

**G2 — Answer the asr research-integrity question. Runs BEFORE any fix.** *(promoted ahead of G1 by Addendum C)*
Read `asr/src/asr/analysis/biophysics.py:66-70,89` and `asr/src/asr/alphabet_reconcile.py:1-11`. Query bathos for runs touching `compute_esmc_site_scores` / `get_fidelity_report`. Include the downstream consumers Addendum C names: `asr/src/asr/analysis/interpret.py:21,71` and `asr/scripts/compare_distributions.py:21,27`.
**Pass:** a written statement of which published asr numbers, if any, change when the alphabet fix lands — recorded *while the current behaviour is still reproducible*. Fixing first destroys the ability to answer this cleanly.

**G-α — Decide, in one sentence, what alphabet a proteinsmc integer sequence is.** *(new gate, Addendum C, C3)*
Neither the plan nor this errata ever makes this decision, and the two halves of the original G1 assumed opposite answers: "apply `af_to_mpnn` before the scorer" presumes sequences are **AF**-ordered; "rederive the ESM map from `MPNN_ALPHABET`" presumes they are **MPNN**-ordered. An executor following both literally will mis-convert or double-convert the ESM path.
**Pass:** a one-line declaration committed to `src/proteinsmc/utils/constants.py` as a module docstring — "proteinsmc integer sequences are &lt;AF|MPNN&gt;-ordered, 21 states, X=&lt;n&gt;" — plus a test asserting it. Every conversion decision downstream, including the ESM map rebuild and G1 below, derives from this line. **This is the true head of the dependency chain; nothing else in the alphabet group is decidable without it.** Note `asr/src/asr/alphabet_reconcile.py:1-11` exists precisely to warn about the failure class this gate closes.

**G1 — Make every cross-alphabet boundary explicit at the call site.** *(rescoped by Addendum C, C1/N2/N3)*
```bash
rg -n 'af_to_mpnn|mpnn_to_af|MPNN_ALPHABET|aa_convert' src/ tests/   # currently exit 1
rg -n 'make_score_fn|remap_sequences|one_hot' ~/projects/asr/src/asr/analysis/biophysics.py
uv run python -c "
restypes='ARNDCQEGHILKMFPSTWYV'; mpnn='ACDEFGHIKLMNPQRSTVWYX'
print('AF==MPNN?', restypes==mpnn[:20])
print('fixed points:', [i for i in range(20) if restypes[i]==mpnn[i]])"   # False; [0, 15, 16]
```
**Pass — all four, not just the first two:**
1. **`asr/src/asr/analysis/biophysics.py:66-70` converts before `make_score_fn`'s returned callable.** This is the live defect. It bypasses `aminx/host/` entirely, so aminx's four internal `af_to_mpnn` sites never fire for asr. *Nothing in the original gate list covered this.*
2. `PROTEINMPNN_TO_ESM_AA_MAP_JAX` is renamed to state its source alphabet, resized to 21, and given an explicit `X → <unk>` — with its **content derived from G-α's declaration**, not unconditionally from `MPNN_ALPHABET`. Dead branches at `constants.py:290-297` deleted.
3. `scoring/mpnn.py` converts before `score_fn` — but note this is **latent, not live**: `mpnn.py:14`'s `PRXTEINMPNN_AVAILABLE = find_spec("prxteinmpnn")` is false, so `make_mpnn_score` raises at `:80-82` and the body never runs. Fix it for when phase −1 makes it reachable; **do not treat this row as evidence the bug is fixed.**
4. `tests/utils/test_translation.py:72`'s `# A, C, X` comment is corrected, and a test covers index 20 reaching `utils/esm.py:52`. Per Addendum C N3, `biophysics.py:74`'s `one_hot(sequence, num_classes=21)` proves the 21st state is reachable in the **live protein path**, not only the codon path this errata originally implied.

**This gate blocks the entire aminx group.** It does not block G2, which must already be done.

**G3 — Move the right copy of `scoring/mpnn.py`.**
```bash
diff -rq --exclude=__pycache__ ~/projects/asr/vendor/proteinsmc/src/proteinsmc \
                               ~/projects/proteinsmc/src/proteinsmc
cd ~/projects/asr/vendor/proteinsmc && git log --oneline -1
```
**Pass:** the single differing file is `scoring/mpnn.py`, the fork is at `9f42237`, and the migration takes the **fork** (deferred imports) as its source. Confirm `aminx/src/aminx/host/prep.py:21`'s module-scope `create_protein_dataset` import does not re-enter the collapsed adapter's import path.

**G4 — Replace the cycle grep with a real test.**
Delete the §2 line 43 / §7 line 470 grep. Add:
```bash
uv run --isolated --with aminx --with-editable . python -c \
 "import proteinsmc, sys; assert not [m for m in sys.modules if m.startswith('aminx')], \
  [m for m in sys.modules if m.startswith('aminx')]"
```
**Pass:** exits 0. Entry-point resolution in `registry.py` must be lazy — never at `proteinsmc/__init__` import time.

**G5 — Rewrite the three `chunked_map` call sites; do not swap the name.**
```bash
rg -n 'chunked_map|safe_map' src/proteinsmc/
python3 -c "print(100 % 64)"                       # 36 -> smc_tree defaults raise under xtrax
```
**Pass:** `mutation.py:217-221` and `memory.py:137,142` take single-arg closures; both `smc_tree.py` `static_args` (`:83-87`, `:144-148`) are closure-captured; `smc_tree.py:290` has an explicit divisibility policy (`num_leaves` is unconstrained); and the empty-pytree and non-divisible cases are decided against `tests/utils/test_jax_utils.py:24-31,34-41,81-90`.

**G6 — Enumerate asr breakage correctly and repoint it in the same phase.**
```bash
cd ~/projects/asr
rg -n "^\s*(from|import)\s+proteinsmc" --glob '*.py' \
   -g '!vendor/**' -g '!**/.venv/**' -g '!**/__pycache__/**' .
ls -la esm_models/                                 # esmc_300m.eqx, 1.33 GB
```
**Pass:** the phase-3 list names **15** statements across 13 files (not 10), including all six ESM sites and `scripts/analyze_oed.py:11-12` + `scripts/diagnose_gp.py:4-5` (which import proteinsmc's `OEDDesign`). The 1.33 GB weights file has a stated destination.

**G7 — Assert the destinations that are contested are not overwritten.**
```bash
ls ~/projects/asr/src/asr/oed/        # __init__ experiment gp padding structs — no tracking.py
ls ~/projects/aminx/src/aminx/        # model/ singular — no models/
rg -c -i -w 'esm|esmc' ~/projects/aminx/src ~/projects/asr/aminx/src   # expect 0
```
**Pass:** `asr/oed/experiment.py` and `asr/oed/structs.py` are untouched; proteinsmc's arrive as `smc_experiment.py` / `smc_design.py`. `utils/esm.py` targets `aminx/model/esmc.py`. `load_model` is namespaced against `aminx/io/weights.py:151`.

**G8 — Do not delete live code.**
```bash
rg -n 'auto_tune_batch_size|AutoTuningConfig' src/proteinsmc/runner.py \
   src/proteinsmc/models/memory.py src/proteinsmc/oed/experiment.py tests/conftest.py
rg -n 'models\.types|models import types' src/ tests/ | wc -l
wc -l src/proteinsmc/sampling/initialization_factory.py    # 393
rg -i 'checkpoint|resume' ~/projects/bathos/src/bathos --type py
rg -l 'arrayrecord|ArrayRecord' ~/projects/bathos          # 0 files
```
**Pass:** `utils/memory.py` + `models/memory.py` are retitled REWRITE and paired; `models/types.py` is scheduled as a codemod; `initialization_factory.py` is retitled SPLIT/REWRITE with the blackjax dispatch and PRSMC constructor explicitly assigned; `oed/tracking.py` is split three ways with the checkpoint and ArrayRecord halves marked "no destination — blocked on Q-owner decision"; `utils/serialization.py` is marked conditional on §3.5.

**G9 — Correct the §7 verification steps that cannot pass as written.**
- Line 476 ("`xtrax plan` reproduces the batch sizes the deleted heuristics chose") — `AxisDecision.batch_size` is `plan.py:96-106` "from spec." Rewrite or drop.
- The §5 codon wall — rewrite as `per-file-ignores` over the complement, note `pyproject.toml:44` excludes `tests`, note `importlib.import_module` bypasses it, and describe it as a smoke alarm.
- §1 line 9 — remove the "aminx structurally cannot host an iterative sampler" argument (`aminx/src/aminx/potts/sampling.py`, 618 LOC, refutes it) and record the two collisions it denies.
- §6.2 line 356 vs §7 line 406 — reconcile "the only implementation" against "4 implementations."
- §8 Q5 — add the jax `<0.11` ceiling and five transitive runtime deps (`xtrax/pyproject.toml:7`).

**G10 — Only then run phase 3.** After the port, re-run G4 and G6; asr should import zero proteinsmc, and `rg 'chunked_map'` should return nothing.

The three gates below were **added by Addendum C** — each closes a hole the original list left open.

**G11 — Restate §1's justification before any row that depends on it executes.** *(Addendum C gate assessment)*
G9 asks the plan to *remove* the false "aminx structurally cannot host an iterative sampler" argument, but nothing gates the move table on the *replacement*. Since the premise is refuted (`aminx/src/aminx/potts/sampling.py`, 618 LOC, imports none of the banned modules) and Addendum A.7 already conceded it, an executor reading top-to-bottom hits the warning banner, then §1's argument, with no gate between them.
**Pass:** §1 argues "proteinsmc keeps the sampler contract" on grounds that survive — the L1-leaf dependency argument, the sampler-contract-is-the-product argument, and the collision matrix — and §6.1's `DECODERS` rename plus the "aminx imports proteinsmc" direction are each re-justified or dropped. **Blocks §6.1 and the framing of §4's aminx group; does not block the individual file rows.**

**G12 — Resolve the exact-vs-range xtrax pin before adopting it.** *(Addendum C, N1)*
```bash
grep -n xtrax ~/projects/aminx/pyproject.toml ~/projects/prolix/pyproject.toml
```
`aminx/pyproject.toml:26` is `xtrax[io]==0.4.0a5` — **double-equals** — while `prolix/pyproject.toml:32` is `xtrax>=0.4.0a5,<0.5`. §8 Q5 asks the author to accept a *range* pin, but in any environment containing aminx (i.e. every asr environment, the only one that runs the full DAG) the effective constraint is already exact. Any xtrax patch release breaks all of L2 simultaneously.
**Pass:** Q5 is rewritten to state the real constraint, and either aminx relaxes to a range or the plan accepts a coordinated-release policy across L2 explicitly. Note the coupling Addendum C N4 identifies: `xtrax/pyproject.toml:7` pins `jax>=0.10.2,<0.11`, which is *also* what keeps `proxide.ops.dataset`'s `jax_md` import working — so the phase-4 xtrax adoption and the phase-3 deferred-import decision are joined through a jax bound neither document connects. If xtrax ever relaxes `<0.11`, the asr fork's rationale reactivates.

**G13 — Decide `oed/opt.py`'s fate; it is entangled and currently degenerate.** *(Addendum C gate assessment)*
B.4 says "do not move — fix or delete," but that verdict lives only in a table with no gate sequencing it, and `opt.py` is coupled to `run.py:108-137 train_surrogate_model` through `tests/oed/test_opt.py:4`.
**Pass:** a written decision recorded for `opt.py:63`'s rank-1 `fim[i,j] = d_info_i * d_info_j` (independently re-confirmed: forward differences of the single scalar `information_gain`, so `det ≡ 0` and `run.py:157`'s `argmax` selects over float noise — **no jitter, regularizer, or alternate path exists in the 68-line file**). Addendum C adds a second degeneracy the errata under-stated: `run.py:95-98` casts `N`/`K`/`q`/`population_size` to `int`, so `opt.py:36`'s ε=1e-4 perturbation of four discrete parameters yields derivatives that are likely *exactly* zero, not merely uninformative. Decide whether the criterion is repaired (a real multi-output Jacobian) or the acquisition is replaced, and say what that means for any BO result already reported.

**G14 — Sequence the interior of phase −1 → phase 0.** *(Addendum C, N5)*
`runner.py:303-304` leaves `final_state = None` for five of six registered samplers and falls through to the write path — a silent no-op. Phase 0 must fix it, but phase 0's verification ("each of the six registered samplers runs ≥3 steps") needs a working environment, which is phase −1's output, while phase −1's contract selection is itself a multi-day exercise. The two phases are gated on each other with nothing sequencing the middle.
**Pass:** phase −1 is decomposed into (a) drop `prxteinmpnn` from `pyproject.toml:20`, (b) `uv sync` green, (c) baseline failure count recorded — with (c) explicitly *not* requiring the six-sampler check, which moves wholly into phase 0.

---

## B.7 Did the plan's method hold up?

**No. The move table needs systematic re-derivation before it is executable, though the plan's architecture does not.**

The failure rate is not marginal. Of 60 audited claims, **41 were refuted or required qualification** — roughly two thirds. More diagnostically, the *kind* of failure is uniform. Every serious error in this document has one of three shapes, and all three are the potts shape:

**Shape 1 — a shared word treated as a shared thing (11 instances).** `PROTEINMPNN_X_INT` is AlphaFold-indexed. `needs_translation` is an integer comparison, not a translation. `oed/experiment.py` names two unrelated pipelines. `oed/structs.py` names two disjoint design spaces. `metrics.py`'s "shannon_entropy" and asr's are `scipy.stats.entropy` under an alias. proteinsmc's "memory" is a wall-clock probe; xtrax's is a byte ceiling. `xtrax.distributed.sharding` is annotation, not execution. prolix's "geometric ladder" is Kelvin ascending where proteinsmc's β is ascending — reciprocal quantities on orthogonal axes. "safe_map" means two different functions inside one file (`smc_tree.py:10` vs the proposed xtrax one). trex's `AxisSpec` is not xtrax's. asr's "Potts order" is not aminx's.

**Shape 2 — a structural signal read as a semantic conclusion (6 instances).** The lint wall at `aminx/pyproject.toml:143-169` read as "aminx cannot host iterative samplers," when aminx ships 618 LOC of exactly that under it — the plan's own §1 premise. A top-level string dispatch read as the whole of a 393-line file. A "legacy redirect" docstring read as deadness across 15 importers. An import count read as separability for `oed/tracking.py`. A one-line `oed/nk.py` read as a type edge when it is a runtime call into a 239-line module.

**Shape 3 — a negative asserted from a search narrower than the claim (5 instances).** "asr has 10 import sites" (15). "13-file codon path" (11 enumerated, ~18 real). "asr has no phase-boundary concept" (searched the function name; asr has a pre-registered K-μ frontier program). "the only implementation with compiler-derived memory budgeting" (five). "4 vocabularies, 3 permutations" (asr excluded; ≥6 and ≥8).

Two things sharpen the verdict.

**First, the method's blind spot is directional, and the audit inherited it.** The plan checked whether code should *leave* far more carefully than whether the destination was right. The audit's own first pass did the same — its destination check for `utils/pmap_utils.py` was, in full, "`sharding.py` is present," and it certified xtrax as owning `padding.py` without reading `bucket.py`'s docstring. That the same blind spot appeared in an audit *written to catch it* is the strongest available evidence that it is a property of the method, not of any one author. Coupling metrics tell you where the edges are. They are structurally silent on whether the thing at the other end is the same kind of thing.

**Second, the method's successes are real and should not be discarded.** The DAG in §2 is correct. `xtrax` genuinely is the right dependency direction, and `xtrax.tiling` genuinely is general (one real cross-domain consumer, and the `xtrax.inference` coupling was deliberately severed — `xtrax/src/xtrax/tiling/roles.py:1-8`). The codon quarantine's *placement* decision survives a clean eight-repo negative search: there is no genetic code table, no codon→AA map, no CAI implementation, no substitution model anywhere else. `oed/` really does belong to asr, and the phase-3 gate rationale ("asr's entire current import surface points at code this phase moves out") is exactly right — all 15 statements hit either `oed/` or ESM. `scoring/mpnn.py` really is the edge-flip file and really is aminx-family by the runtime-instantiation test. §3.1's remedy (`eqx.filter_jit`) is right even though its stated evidence is a non-sequitur. Roughly a dozen DELETE rows are clean.

**Direct answer.** Do not execute the move table as written. Re-derive it under one rule the plan never applied: **for every row, read the destination's source and state what makes the two things the same kind of thing** — the class docstring, the reference architecture, what the module instantiates at runtime, where its weights come from, and what alphabet or unit convention it uses. That is a per-row cost of perhaps twenty minutes and it would have caught all 41.

Two rows are not merely wrong but actively harmful and must be fixed before any code moves: the alphabet delete (§B.2.1), which would relocate a live silent-corruption bug into the one repo whose `aa_convert.py` exists to prevent it; and §1's aminx premise (§B.3.17), which is the potts error sitting in the plan's foundation, where every downstream row inherits it.

---

# Addendum C — Independent Final Review

**Reviewer:** Opus 5, read-only, dispatched 2026-08-13 after Addendum B was published.
**Brief:** verify the combined plan+errata artifact against source, on the explicit instruction *not* to take either document's claims on faith.
**Verdict: NEEDS ANOTHER PASS** — "the errata is substantially correct and its architecture-survives / move-table-fails split holds, but its own lead gate (G1) prescribes a fix to a dead code path while leaving the live bug it discovered unfixed, and G1/G2 are ordered backwards relative to the errata's own reasoning — so executing the combined artifact 'respecting the gates' would still ship the corruption."

All corrections below have been **folded into B.1–B.6 in place**, each marked inline. This section is the provenance record and the list of findings that were new.

## C.1 What was corrected in Addendum B

| # | Section | Correction |
|---|---|---|
| C1 | B.6 G1 | **The lead gate fixed dead code.** `scoring/mpnn.py:14` gates on `find_spec("prxteinmpnn")`, which does not resolve, so `make_mpnn_score` raises at `:80-82` and never executes. B knew this (B.2.11) and still made it the gate. The live corruption is at `asr/analysis/biophysics.py:66-70`, which G1 never mentioned. G1 rescoped; see also N2. |
| C2 | B.6 G1/G2 | **Gates ordered backwards against B's own reasoning.** B.5 item 3 says the integrity question must be answered *before* the fix changes the numbers; the list ran fix-first. Reordered to G0 → G2 → G-α → G1. |
| C3 | B.2.1, B.6 | **The prescribed fix was internally inconsistent about which alphabet is canonical.** G1 presumed AF-ordered sequences; the same section's table presumed MPNN. If sequences are AF the ESM map's *content* is already correct and rebuilding from `MPNN_ALPHABET` would break it. The load-bearing question — what alphabet is a proteinsmc sequence? — was never decided in either document. New gate **G-α** added to force the decision first. |
| C4 | B.1 | **Decode fixed-point count wrong: 5 of 24, not 1.** The decoded string `LTDVHIWWWQAQQWQHSAITGFIG` recomputed exact; the permutation has three fixed points (A, S, T at indices 0/15/16) and the peptide contains five such residues. Finding unaffected, headline number was wrong. |
| C5 | B.2.11 (no change needed) | The reviewer **agrees** with B against the plan: Python closures hash by identity, so a closure over `jnp` arrays *is* a valid `static_argnames` value. §3.1's "does not compile today" is a non-sequitur; `eqx.filter_jit` remains right as a precondition of the *proposed* `eqx.Module` `Fitness`, not a fix for a live failure. |
| C6 | B.1, B.2.1 | **Line-level slips in the lead finding.** The lint block has **sixteen** entries (four potts + twelve unrelated `xtrax.*` rules), not "exactly four modules." B.2.1 transposed `aa_convert.py:16`/`:17` and contradicted B.1. `AMINO_ACIDS_NUM_STATES` is `:119`, not `:118`. |
| C7 | B.3.16 | **Internal contradiction on trex.** Listed among repos with zero `codon|nucleotide` hits; trex returns 5 hits across 2 files, one of which the same section reports below as an unrecorded collision. Conclusion survives, evidence did not. |
| C8 | B.1 | **The 60-claim ledger is not auditable.** Counts sum, but §B.2 has 17 entries vs 14 claimed refuted, §B.3 has 23 vs 27, the 17 survivors are never listed, and "72 raw" appears nowhere. Caveat added. |
| C9 | plan phase −1 | `prxteinmpnn` is at `pyproject.toml:20`, not `:21` — matters because phase −1 is a line-targeted edit. |

## C.2 Findings neither document caught

- **N1 — aminx pins xtrax *exactly*.** `aminx/pyproject.toml:26` is `xtrax[io]==0.4.0a5`; `prolix/pyproject.toml:32` is a range. §8 Q5's tradeoff analysis is wrong about what is being accepted. → gate **G12**.
- **N2 — the asr MPNN path needs a conversion no gate asked for.** `biophysics.py:18` calls `aminx.scoring.score.make_score_fn` directly, bypassing `aminx/host/` and its four `af_to_mpnn` sites. G3 checks the right *copy* moves; G6 enumerates asr's imports; neither covers the direct call where the corruption occurs. → folded into **G1**.
- **N3 — index 20 is reachable in the live asr protein path.** `biophysics.py:74`'s `one_hot(sequence, num_classes=21)` proves a 21st state is legal input today, so the length-20 ESM map's Valine clamp is not confined to the codon path as B framed it. → folded into **G1**.
- **N4 — the xtrax pin collides with the reason the asr fork exists.** `xtrax/pyproject.toml:7` pins `jax>=0.10.2,<0.11`, which is also what keeps `proxide.ops.dataset`'s `jax_md` import working — coupling phase 4 to phase 3 through a bound neither document connects. → folded into **G12**.
- **N5 — phase −1 and phase 0 are gated on each other** with a multi-day contract-selection exercise in between and no interior sequencing. → gate **G14**.

## C.3 Independently re-confirmed

Verified against source, not against Addendum B: the AF ordering at `constants.py:10-31` and the three mis-named constants; `PROTEINMPNN_X_INT = 21` matching neither alphabet (X is 20 in both `aa_convert.py:16` and `:17`); the length-20 gather clamping to `restypes[19] == "V"`; the two dead `"X" in AA_CHAR_TO_INT_MAP` branches; zero alphabet conversion anywhere in `scoring/mpnn.py`; §1's premise false with `potts/sampling.py` at 618 LOC importing none of the banned modules; `opt.py`'s rank-1 FIM with no jitter or alternate path across the full 68-line file; all five DELETE rows reaching live code (`initialization_factory.py` at 393 lines, `models/types.py` with 20 referencing files of which 5 are module-level runtime imports); the L2→L0 dependency direction in both `prolix` and `aminx`; the codon quarantine on a re-run eight-repo negative; and `asr/vendor/proteinsmc` at `9f42237` on `fix/asr-3713-lazy-mpnn-import` with exactly one differing file. Also re-confirmed: `runner.py:303-304`'s silent no-op, `smc_tree.py`'s 100 % 64 = 36 divisibility failure against `xtrax/transforms/map.py:32-37`, the inverted single-arg/tuple `map` convention, and `xtrax/tiling/bucket.py`'s host-side-NumPy character.

## C.4 Residual risk if executed while respecting the gates

1. **Highest:** the corruption ships anyway if the *original* G1 is used — it passes on a function that cannot run while `asr/analysis/biophysics.py` keeps producing mismatched ESM/MPNN site scores through `interpret.py:71` and `compare_distributions.py:27`. Closed by the rescoped G1 + G-α.
2. The canonical alphabet gets decided implicitly by whoever writes the code. Closed by G-α.
3. Published asr numbers change with no record of which. Closed by promoting G2.
4. The partition executes on a justification already refuted. Closed by G11.
5. **Still open:** the move table is not executable row-by-row. B.7's remedy has not been applied to the rows B did not audit — §4 has 66 file rows plus §6.6's collision table against 60 audited claims, and the unaudited remainder is unquantified in both documents. **This is what the re-derivation pass addresses; see Addendum D.**
6. **Still open:** phase −1 → phase 0 interior sequencing beyond what G14 states.

## C.5 A note on the method, extending B.7

B.7 argued its blind spot is a property of the method rather than any one author, citing its own one-line `pmap_utils` destination check as evidence. This review is a third data point in the same direction, and a sharper one: the errata's failures are no longer *destination* errors — it checked destinations well — but **remedy** errors. It correctly identified a live bug and then prescribed a fix to a code path that cannot execute, in a gate list ordered against its own stated reasoning, resting on a canonical-alphabet decision it never made. The lesson generalises past "read the destination": **a gate is not verified until you have confirmed the code it targets actually runs.**