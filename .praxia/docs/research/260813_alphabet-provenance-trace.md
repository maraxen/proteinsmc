# Alphabet Provenance Trace — answering G-α empirically, and scoping G2

**Task:** `260813_proteinsmc_ecosystem_partition`
**Status:** complete for the ESM/MPNN validation path. Read-only; no code changed.
**Supersedes:** the mechanism stated in Addendum B §B.1 and in Addendum C §C1/N2.

---

## Headline

**The broken path is the ESM one, not the MPNN one. Both prior addenda had it backwards.**

asr declares a canonical alphabet, and it is **ProteinMPNN order with gap last**:

```
asr/src/asr/alphabet.py:6-9
    # Canonical Alphabet: Standard 20 AAs + Gap at the end
    # Based on ProteinMPNN's alphabet
    CANONICAL_ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"
```

Given canonical-ordered input at `asr/src/asr/analysis/biophysics.py:89`:

| path | receives | expects | result |
|---|---|---|---|
| `compute_mpnn_site_scores` → `aminx.scoring.score.make_score_fn` | MPNN-ordered ints | MPNN order | **correct** for the 20 amino acids |
| `compute_esmc_site_scores` → `proteinsmc.utils.esm.remap_sequences` | MPNN-ordered ints | **AlphaFold** order (`utils/constants.py:10-31`) | **permuted — every non-fixed-point residue scores as a different amino acid** |

Addendum B asserted that proteinsmc's AF ints were reaching an MPNN-ordered scorer. The
data flows the other way: asr's MPNN-ordered ints reach proteinsmc's AF-ordered ESM table.
The direction of the error is reversed, and so is which metric is contaminated.

`PROTEINMPNN_TO_ESM_AA_MAP_JAX` is therefore **correctly named and incorrectly built** —
its callers genuinely need an MPNN→ESM map; its construction iterates `AA_CHAR_TO_INT_MAP`
= `restype_order` = AF (`constants.py:272-282`). Addendum B §B.2.1 called it "an AF→ESM map
wearing an MPNN name" and prescribed renaming it to match its content. **Do the opposite:
keep the name and fix the content.** Renaming it would enshrine the bug.

## The trace

1. `biophysics.py:89` takes `ancestor_seq` from one of two callers, both of which
   `argmax` a soft posterior over a 21-state axis:
   - `asr/src/asr/analysis/interpret.py:26,68,71` — `np.argmax(self.ancestors, axis=-1)`
   - `asr/scripts/compare_distributions.py:15,27` — `np.argmax(ancestors_soft, axis=-1)`
2. That axis is the model's output vocabulary. asr's canonical is `alphabet.py:8`
   (MPNN + gap at 20), and the repo carries machinery to enforce it:
   `alphabet.py:15-16` (`MPNN_TO_CANONICAL`, `CANONICAL_TO_MPNN`) and
   `alphabet_reconcile.py`, which is imported by 10+ modules including
   `felsenstein_map.py`, `iqtree_runner.py` and `esr_harness.py`.
3. **`biophysics.py` uses none of it.** It imports `remap_sequences` from proteinsmc
   directly (`:7`) and passes `sequence` unconverted to both scorers (`:31`, `:68-70`).
   The one module that mixes two foreign scorers is the one module that bypasses the
   reconciler written to prevent exactly this.

## Why nobody saw it

`biophysics.py:78-79` slices both operands to the first 20 states:

```python
log_probs  = jax.nn.log_softmax(logits, axis=-1)[..., :20]
site_scores = -(seq_one_hot[..., :20] * log_probs).sum(-1)
```

A gap at index 20 becomes an all-zero one-hot row after the slice, so it contributes
**exactly 0.0** to the MPNN score rather than raising. In the ESM path the same gap indexes
a length-20 table (`constants.py:272-276`) and clamps to `restypes[19] = 'V'`
(`utils/esm.py:52`). Both failure modes are numerically silent. This confirms Addendum C's
N3 — index 20 is reachable in the live protein path via `one_hot(..., num_classes=21)` at
`biophysics.py:74` — and supplies the reason it never surfaced.

## asr has four live conventions, not one

The declared canonical is not universally applied. Enumerated across `src/` and `scripts/`:

| ordering | gap | files |
|---|---|---|
| `ACDEFGHIKLMNPQRSTVWY-` (canonical/MPNN) | last | `alphabet.py:8`, `sp2_2_iqtree_map_baseline.py:42`, `sp2_2_pdz_baselines.py:42`, `sp5_4_dms_fitness_eval.py:47`, `validate_pdz_dms.py:17`, `validate_ancestors_continuous.py:30` |
| `ARNDCQEGHILKMFPSTWYV-` (AlphaFold/LG) | last | `esr_validation.py:34`, `esr_benchmark.py:46`, `compare_trex_vs_iqtree.py:27`, `run_trex_influenza.py:38`, `run_mpnn_loss_influenza.py:34`, `analyze_influenza_dca.py:34,205`, `validate_antigenic_sites.py:30,96` |
| `-ACDEFGHIKLMNPQRSTVWY` (Potts) | **first** | **`src/asr/pdz_utils.py:8`**, `compare_thermodynamics.py:30`, `visualize_pdz_results.py:44` |
| `ARNDCQEGHILKMFPSTWYV` (LG, 20-state) | none | `src/asr/jtt_model.py:47`, `src/asr/lg_model.py:17`, `analyze_uncertainty_decomposability.py:24`, `visualize_antigenic_uncertainty.py:38` |

The gap-first Potts convention is in **production source** (`pdz_utils.py:8`) and offsets
every amino acid by one relative to the other two 21-state orders. `alphabet_reconcile.py:3-8`
names three of these four as a known hazard and states the consequence precisely:
*"A silent permutation between these orders produces ARBITRARY recovery-Hamming numbers with
no error signal."* The machinery is right; its coverage is incomplete.

## G-α, answered

**For asr, the canonical alphabet is already decided and documented: `alphabet.py:8`,
MPNN order, gap at index 20.** G-α does not need a new decision on asr's side — it needs
proteinsmc's ESM table rebuilt against that existing declaration, and `biophysics.py`
routed through `alphabet_reconcile.py` like its ten sibling modules.

**For proteinsmc, the question stands but is now narrower.** Its own encoder is AF-ordered
(`constants.py:10-31,113`). Nothing in this trace establishes what a proteinsmc *sampler*
run carries, only what asr feeds into proteinsmc's ESM helper. Declaring proteinsmc AF-ordered
and converting at the boundary remains the minimal change; that decision is unblocked by
this trace but not made by it.

## G2 — scope of affected results

Metrics computed through the **permuted ESM path**, and therefore suspect:

- `avg_esm_pll` — `biophysics.py:102`
- `avg_confidence` and `site_confidence` — `:99,104,105`, both derived from `esm_pll`
- the rollups at `interpret.py:91,93` (`avg_esm_pll`, `avg_confidence`) and their CSV export at `:114-116`
- the ESM column of `compare_distributions.py:35-37`

Metrics that appear **unaffected**, because the MPNN path receives the ordering it expects:

- `avg_mpnn_nll` — `biophysics.py:103`, `interpret.py:92`, `compare_distributions.py:39-41`
- `site_mpnn` — `:106`

This is a materially smaller blast radius than Addendum B implied, and it is the opposite
half of the report. **Any figure or table quoting ESM pseudo-log-likelihood or the
sigmoid-calibrated "confidence" from this validator should be treated as unverified until
the map is rebuilt and the numbers re-run.** Which of those reached a paper, deck or
milestone is not determinable from the code and is the owner's call — that is the part of
G2 this trace cannot close.

## Caveats

- **Nothing here was executed.** proteinsmc cannot be imported (`pyproject.toml:20` declares
  the unresolvable `prxteinmpnn`), so this is a static trace, exactly as with all 82 rows of
  Addendum D. The permutation is read off the alphabet declarations and the absence of any
  conversion at `biophysics.py:31,68-70`; it has not been demonstrated by running the code.
- **The artifact-producing pipeline was not traced to its origin.** The conclusion rests on
  asr's *declared* canonical (`alphabet.py:8`) governing the `ancestors` arrays that
  `interpret.py:12` and `compare_distributions.py:11` load from `.npz`. A producer writing
  in LG or Potts order instead would change which path is wrong — but not the finding that
  one of them is, since `biophysics.py` converts for neither.
- The four-convention census covers `src/` and `scripts/`; notebooks and `vendor/` were not
  enumerated.
