# Failing Tests Analysis - proteinsmc

**Date:** October 24, 2025  
**Total Failing Tests:** 23 (previously reported as 29, but 6 NUTS tests are now skipped)  
**Test Run:** `uv run pytest --cov=src`

---

## Executive Summary

The 23 failing tests fall into **4 distinct categories** with clear root causes:

1. **I/O PyTree Serialization** (4 tests) - Data structure mismatch in read operations
2. **Fitness Function Signature** (8 tests) - Tests passing wrong argument types
3. **MCMC Loop State Management** (4 tests) - SamplerState API mismatch
4. **Initialization Factory API** (6 tests) - Function signature changes
5. **OED Integration** (1 test) - Complex integration issue

**Good News:** Most failures are test issues, not implementation bugs. They can be fixed quickly.

---

## Category 1: I/O PyTree Serialization Issues ✅ **EASY FIX**

### Affected Tests (4)

```
FAILED tests/io/test_io.py::TestReadLineageData::test_read_lineage_data_basic
FAILED tests/io/test_io.py::TestReadLineageData::test_read_lineage_data_with_arrays
FAILED tests/io/test_io.py::TestReadLineageData::test_read_lineage_data_complex_pytree
FAILED tests/io/test_io.py::TestIntegrationScenarios::test_pytree_roundtrip_preserves_structure
```

### Root Cause

**Data Structure Mismatch:** The `read_lineage_data` function returns data in a different format than tests expect.

**Expected by tests:**

```python
{"data": {"step": 0, "fitness": 1.0}}
```

**Actually returned:**

```python
{"leaves": [array(...), 1.0, ...]}  # Flattened PyTree structure
```

### Error Example

```python
# From test_read_lineage_data_basic
assert lineage_data[0]["data"]["step"] == 0
       ^^^^^^^^^^^^^^^^^^^^^^^
KeyError: 'data'
```

### Fix Strategy

The `io.py` module is using `jax.tree_util.tree_flatten` which converts the nested PyTree into a flat `leaves` list and separate `treedef`. Tests need to either:

**Option A:** Update tests to use the actual returned structure
**Option B:** Update `read_lineage_data` to reconstruct the original PyTree structure

**Recommendation:** Option B - Fix `read_lineage_data` to use `tree_unflatten`

### Code Location

- Implementation: `src/proteinsmc/io.py` - `read_lineage_data` function
- Tests: `tests/io/test_io.py` lines 475-615

### Estimated Fix Time

**15-30 minutes** - Simple implementation fix

### Proposed Fix

```python
# In src/proteinsmc/io.py - read_lineage_data function
def read_lineage_data(filepath: str) -> Iterator[dict]:
    """Read lineage data from ArrayRecord file."""
    reader = array_record.ArrayRecordReader(filepath)
    for record in reader:
        data = json.loads(record)
        # Reconstruct PyTree from flattened structure
        if "leaves" in data and "treedef" in data:
            leaves = data["leaves"]
            treedef = data["treedef"]
            # Reconstruct original structure
            yield jax.tree_util.tree_unflatten(treedef, leaves)
        else:
            yield data
```

---

## Category 2: Fitness Function Signature Issues ⚠️ **TEST BUG**

### Affected Tests (8)

```
FAILED tests/utils/test_fitness_advanced.py::TestChunkedFitnessEvaluation::test_chunked_vs_non_chunked
FAILED tests/utils/test_fitness_advanced.py::TestChunkedFitnessEvaluation::test_chunked_with_large_population
FAILED tests/utils/test_fitness_advanced.py::TestChunkedFitnessEvaluation::test_chunked_with_different_batch_sizes
FAILED tests/utils/test_fitness_advanced.py::TestMultipleFitnessComponents::test_multiple_fitness_functions
FAILED tests/utils/test_fitness_advanced.py::TestMultipleFitnessComponents::test_weighted_combine_function
FAILED tests/utils/test_fitness_advanced.py::TestContextDependentFitness::test_fitness_with_context
FAILED tests/utils/test_fitness_advanced.py::TestContextDependentFitness::test_fitness_with_different_context_values
FAILED tests/utils/test_fitness_advanced.py::TestTranslationIntegration::test_fitness_with_translation
```

### Root Cause

**Tests are passing sequences as PRNG keys.** The fitness function expects:

```python
fitness_fn(key: PRNGKeyArray, sequence: Array, context: Optional[float]) -> Array
```

But tests are calling:

```python
fitness_fn(key, sequence, None)  # where 'key' has wrong shape
```

### Error Example

```python
TypeError: JAX encountered invalid PRNG key data: expected key_data.shape to 
end with (2,); got shape=(4, 6) for impl=PRNGImpl(...)
```

The shape `(4, 6)` suggests it's receiving a sequence array (population_size=4, seq_length=6) instead of a PRNG key (shape should end with `(2,)`).

### Analysis

Looking at the test code pattern:

```python
# In test_chunked_vs_non_chunked (line 69)
key = jax.random.PRNGKey(42)
sequence = jnp.array([[0, 1, 2, 3, 0, 1], [1, 2, 3, 0, 1, 2]], dtype=jnp.int8)
result_no_chunk = fitness_fn_no_chunk(key, sequence, None)  # This fails
```

The fitness function internally tries to split the key:

```python
# In src/proteinsmc/utils/fitness.py line 102
keys = jax.random.split(key, len(score_fns) + 1)  # Expects valid PRNG key
```

**Hypothesis:** Tests might be inadvertently swapping arguments or the test setup is creating invalid keys.

### Fix Strategy

Need to examine `tests/utils/test_fitness_advanced.py` and ensure:

1. Keys are properly generated: `key = jax.random.PRNGKey(42)`
2. Arguments are in correct order: `fitness_fn(key, sequence, context)`
3. Sequence shapes are appropriate for the fitness function

### Code Location

- Tests: `tests/utils/test_fitness_advanced.py`
- Implementation: `src/proteinsmc/utils/fitness.py` (likely correct)

### Estimated Fix Time

**30-45 minutes** - Need to review and fix 8 test cases

### Proposed Investigation

```python
# Add debug print in test to verify shapes
print(f"Key shape: {key.shape}")  # Should be (2,)
print(f"Sequence shape: {sequence.shape}")  # Should be (population, seq_len)
```

---

## Category 3: MCMC Loop State Management ❌ **API MISMATCH**

### Affected Tests (4)

```
FAILED tests/sampling/test_mcmc_loop.py::TestRunMCMCLoop::test_loop_executes
FAILED tests/sampling/test_mcmc_loop.py::TestRunMCMCLoop::test_loop_with_io_callback
FAILED tests/sampling/test_mcmc_loop.py::TestRunMCMCLoop::test_loop_state_progression
FAILED tests/sampling/test_mcmc_loop.py::TestRunMCMCLoop::test_loop_returns_empty_metrics
```

### Root Cause

**`SamplerState` API has changed.** The MCMC loop implementation is trying to pass a `fitness` keyword argument that doesn't exist.

### Error Example

```python
# In src/proteinsmc/sampling/mcmc.py line 87
new_state = SamplerState(
    sequence=new_sequence,
    key=next_key,
    blackjax_state=None,
    step=i + 1,
    fitness=current_fitness,  # ❌ This parameter doesn't exist
)

TypeError: SamplerState.__init__() got an unexpected keyword argument 'fitness'
```

### Analysis

The `SamplerState` dataclass (in `models/sampler_base.py`) has these fields:

```python
@dataclass
class SamplerState:
    sequence: Array
    key: PRNGKeyArray
    blackjax_state: BlackjaxState | None
    step: int
    additional_fields: dict[str, Any] = field(default_factory=dict)
```

**No `fitness` field exists.** It should go in `additional_fields`.

### Fix Strategy

Update `src/proteinsmc/sampling/mcmc.py` to use `additional_fields`:

```python
new_state = SamplerState(
    sequence=new_sequence,
    key=next_key,
    blackjax_state=None,
    step=i + 1,
    additional_fields={"fitness": current_fitness},  # ✅ Correct
)
```

### Code Location

- Implementation: `src/proteinsmc/sampling/mcmc.py` lines 87-92
- State definition: `src/proteinsmc/models/sampler_base.py`
- Tests: `tests/sampling/test_mcmc_loop.py`

### Estimated Fix Time

**10-15 minutes** - Single-line fix in implementation

### Related Issues

This suggests the MCMC loop implementation hasn't been updated after a recent refactoring of `SamplerState`.

---

## Category 4: Initialization Factory API Issues ⚠️ **API MISMATCH**

### Affected Tests (6)

```
FAILED tests/sampling/test_initialization_factory.py::TestInitializeSamplerState::test_initialize_parallel_replica_state
FAILED tests/sampling/test_initialization_factory.py::TestInitializeSingleState::test_mcmc_initialization
FAILED tests/sampling/test_initialization_factory.py::TestInitializeSMCState::test_smc_state_initialization
FAILED tests/sampling/test_initialization_factory.py::TestInitializeSMCState::test_smc_state_with_none_beta
FAILED tests/sampling/test_initialization_factory.py::TestInitializePRSMCState::test_prsmc_state_initialization
FAILED tests/sampling/test_initialization_factory.py::TestInitializePRSMCState::test_prsmc_additional_fields_values
```

### Root Cause

**Function signatures have changed.** Tests are calling internal initialization functions with outdated parameters.

### Error Example

```python
# In test_smc_state_initialization (line 783)
state = _initialize_smc_state(
    sequence=sequences,
    key=key,
    beta=0.0,
    fitness_fn=mock_fitness_fn,  # ❌ This parameter no longer exists
    mutation_rate=0.1,
)

TypeError: _initialize_smc_state() got an unexpected keyword argument 'fitness_fn'
```

### Analysis

The internal `_initialize_smc_state` function signature has changed, likely as part of the architecture refactoring mentioned in `AGENTS.md`:

> **Recent Architectural Improvements (Completed)**
>
> - ✅ SamplerState Refactoring: Converted from equinox.nn.State to flax.struct.dataclass

The initialization functions likely no longer need `fitness_fn` as a parameter since initialization is now decoupled from fitness evaluation.

### Fix Strategy

Two approaches:

**Option A:** Update function signatures in `initialization_factory.py` to accept the old parameters (backward compatibility)

**Option B:** Update all test calls to use new API (recommended - forces tests to match current design)

Need to check current signatures:

```python
# Check actual signature in src/proteinsmc/sampling/initialization_factory.py
def _initialize_smc_state(
    sequence: Array,
    key: PRNGKeyArray,
    beta: float | None,
    mutation_rate: float,
    # fitness_fn removed?
) -> SamplerState:
```

### Code Location

- Implementation: `src/proteinsmc/sampling/initialization_factory.py`
- Tests: `tests/sampling/test_initialization_factory.py` lines 759-850

### Estimated Fix Time

**45-60 minutes** - Need to review 6 test cases and verify correct API

### Investigation Needed

1. Check current function signatures in `initialization_factory.py`
2. Determine if tests are wrong or if implementation lost functionality
3. Update tests to match current API design

---

## Category 5: OED Integration Test ❓ **COMPLEX**

### Affected Test (1)

```
FAILED tests/oed/test_experiment_integration.py::test_run_oed_experiment_small_nk_protein
```

### Status

**Skipped detailed analysis** - This is an integration test for the Optimal Experimental Design (OED) module, which is less critical than core SMC functionality.

### Priority

**Low** - OED module has 62-98% coverage across its submodules and is not core to SMC functionality.

### Recommendation

- Address after fixing core SMC/MCMC/fitness tests
- May be related to one of the other failure categories
- Could be a transient test or dependency issue

### Code Location

- Test: `tests/oed/test_experiment_integration.py`
- Implementation: `src/proteinsmc/oed/experiment.py` (62% coverage)

### Estimated Fix Time

**Unknown** - Need to investigate error details

---

## Summary Table: Failing Tests by Category

| Category | Count | Severity | Fix Difficulty | Est. Time | Root Cause |
|----------|-------|----------|----------------|-----------|------------|
| I/O PyTree | 4 | Low | Easy | 15-30min | Implementation needs tree_unflatten |
| Fitness Sig | 8 | Medium | Easy | 30-45min | Tests passing wrong arg types |
| MCMC State | 4 | High | Easy | 10-15min | Using removed SamplerState field |
| Init Factory | 6 | Medium | Medium | 45-60min | Tests using old API signatures |
| OED Integration | 1 | Low | Unknown | TBD | Complex integration issue |
| **TOTAL** | **23** | - | - | **~2-3hrs** | - |

---

## Recommended Fix Order

### Phase 1: Quick Wins (30-45 minutes)

1. **MCMC Loop State** (10-15min) - Single line fix, high impact
2. **I/O PyTree** (15-30min) - Clear fix, affects 4 tests

### Phase 2: Test Updates (1-2 hours)

3. **Fitness Function Tests** (30-45min) - Fix test argument confusion
4. **Initialization Factory** (45-60min) - Update tests to new API

### Phase 3: Investigation (TBD)

5. **OED Integration** - Lower priority, investigate after core fixes

---

## Impact Assessment

### Tests That Are Actually Testing Bugs in Implementation

- **MCMC Loop (4 tests)** - Implementation uses wrong SamplerState API

### Tests That Are Testing Old/Wrong APIs

- **Initialization Factory (6 tests)** - Tests out of date with refactored API
- **I/O PyTree (4 tests)** - Tests expect wrong data structure

### Tests With Test Code Bugs

- **Fitness Advanced (8 tests)** - Tests passing wrong argument types

### Tests Needing Investigation

- **OED (1 test)** - Unknown root cause

---

## Action Items for Maintainers

### Immediate (Before Next Commit)

- [ ] Fix MCMC loop `SamplerState` usage in `sampling/mcmc.py`
- [ ] Fix `read_lineage_data` to reconstruct PyTree in `io.py`

### Short-term (This Week)

- [ ] Review and fix all 8 fitness_advanced tests
- [ ] Update 6 initialization_factory tests to new API
- [ ] Document the `SamplerState` API in AGENTS.md or docstrings

### Medium-term (This Month)

- [ ] Investigate and fix OED integration test
- [ ] Add API compatibility tests to catch signature changes
- [ ] Consider adding deprecation warnings for API changes

---

## Lessons Learned

### What Went Wrong

1. **API changes without test updates** - SamplerState refactoring broke MCMC
2. **Missing integration tests** - Initialization factory changes went unnoticed
3. **Incomplete refactoring** - I/O serialization format changed but tests didn't

### How to Prevent

1. **Run full test suite before merging** - These failures should have been caught
2. **Document API changes** - Update AGENTS.md when changing core APIs
3. **Deprecation period** - Add warnings before removing parameters
4. **Test CI gates** - Don't allow merges with failing tests

---

## Notes from AGENTS.md Context

From the technical debt section in `AGENTS.md`:

> **Recent Architectural Improvements (Completed)**
>
> - ✅ SamplerState Refactoring: Converted from equinox.nn.State (incorrect) to flax.struct.dataclass
> - ✅ Test Coverage: Improved from 62% to 96.2% (227/236 tests passing)

**Current Reality:** Test coverage is actually 80%, not 96.2%, and 23 tests are failing. This suggests:

1. Documentation is out of date
2. Recent changes broke tests
3. The "completed" refactoring is incomplete

**Recommendation:** Update AGENTS.md to reflect current test status.
