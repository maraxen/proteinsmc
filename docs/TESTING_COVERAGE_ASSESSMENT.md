# Testing Coverage Assessment for proteinsmc

**Date:** October 24, 2025  
**Overall Coverage:** 80% (2194 statements, 439 missed)  
**Test Status:** 310 passed, 29 failed, 10 skipped

---

## Executive Summary

The codebase has **good overall coverage at 80%**, but there are **critical gaps** in several key areas:

### Critical Gaps (High Priority)

1. **Protocol definitions** (`models/protocols.py`): **0% coverage** - No tests exist
2. **Type mutation protocols** (`models/mutation.py`): **0% coverage** - Core mutation interfaces untested
3. **Translation protocols** (`models/translation.py`): **0% coverage** - Type definitions untested
4. **JAX UUID utilities** (`utils/jax_uuid.py`): **0% coverage** - Utility functions untested
5. **Pmap utilities** (`utils/pmap_utils.py`): **35% coverage** - Parallel mapping poorly tested

### Medium Priority Gaps

6. **ESM integration** (`utils/esm.py`): **47% coverage** - Large module (215 statements), half untested
7. **Fitness evaluation** (`utils/fitness.py`): **62% coverage** - Core fitness logic has gaps
8. **Metrics calculation** (`utils/metrics.py`): **66% coverage** - Statistical functions partially untested
9. **Annealing schedules** (`utils/annealing.py`): **62% coverage** - Important for SMC convergence
10. **Memory management** (`models/memory.py`): **68% coverage** - Critical for performance

### Sampler Loop Gaps (Implementation Issues)

11. **HMC loop** (`sampling/hmc.py`): **36% coverage** - 5 tests skipped, needs refactoring
12. **NUTS loop** (`sampling/nuts.py`): **37% coverage** - 6 tests failing, dtype issues
13. **Gibbs loop** (`sampling/gibbs.py`): **48% coverage** - Partially tested
14. **MCMC loop** (`sampling/mcmc.py`): **87% coverage but 4 tests failing** - Integration issues

---

## Detailed Coverage Analysis

### 1. Core SMC Implementation ✅ **WELL COVERED**

**`sampling/particle_systems/smc.py`**: **95% coverage** (56 statements, 3 missed)

**Tests:** 9 tests, all passing

- ✅ `test_resample_output_shape_and_type` (4 resampling approaches)
- ✅ `test_resample_invalid_approach`
- ✅ `test_create_smc_loop_func_base_smc`
- ✅ `test_create_smc_loop_func_not_implemented`
- ✅ `test_run_smc_loop_basic`
- ✅ `test_run_smc_loop_end_to_end`

**Missing Coverage:**

- Lines likely related to edge cases in `scan_body` function
- Potentially untested error paths in `create_smc_loop_func`

**Recommendation:** Add tests for:

- Different `SMCAlgorithmType` variants beyond "BaseSMC" and "AnnealedSMC"
- Edge cases in annealing function handling
- Error handling when `writer_callback` fails
- Verification of `io_callback` invocations

---

### 2. Parallel Replica SMC ✅ **EXCELLENT COVERAGE**

**`sampling/particle_systems/parallel_replica.py`**: **100% coverage** (109 statements)

**Tests:** 5 tests, all passing

- ✅ `test_migrate_basic_functionality`
- ✅ `test_run_prsmc_loop_basic`
- ✅ `test_run_prsmc_loop_end_to_end`
- ✅ `test_migration_info_creation`
- ✅ `test_prsmc_output_creation`

**Status:** This is the gold standard for test coverage in the project.

---

### 3. **CRITICAL GAP:** Protocol Definitions ❌ **NO COVERAGE**

**`models/protocols.py`**: **0% coverage** (8 statements, all missed)

**Current Tests:** None exist

**Why Critical:**

- Defines core interfaces like `MutationFn` that the entire codebase relies on
- Type checking depends on these protocols
- No validation that implementations actually match protocols

**Recommendation:** Create `tests/models/test_protocols.py` with:

```python
def test_mutation_fn_protocol_compliance():
    """Verify implementations match MutationFn protocol."""
    
def test_fitness_fn_protocol_compliance():
    """Verify implementations match FitnessFn protocol."""
    
def test_protocol_type_hints():
    """Ensure protocols have correct type annotations."""
```

---

### 4. **CRITICAL GAP:** Mutation Type Definitions ❌ **NO COVERAGE**

**`models/mutation.py`**: **0% coverage** (5 statements, all missed)

**Current Tests:** None exist

**Why Critical:**

- Defines `MutationFn` type alias used throughout SMC
- Core to the mutation step in all samplers
- No tests verify the type structure

**Recommendation:** Create basic validation tests for type definitions.

---

### 5. **CRITICAL GAP:** Translation Protocols ❌ **NO COVERAGE**

**`models/translation.py`**: **0% coverage** (5 statements, all missed)

**Current Tests:** None exist (though `test_translation.py` exists for utils)

**Why Critical:**

- Defines `TranslateFuncSignature` protocol
- Critical for nucleotide ↔ protein conversions
- No validation of protocol compliance

---

### 6. Fitness Evaluation **MODERATE GAPS** ⚠️

**`utils/fitness.py`**: **62% coverage** (53 statements, 20 missed)

**Current Tests:**

- ✅ Basic function creation tests pass
- ❌ **8 advanced tests failing** in `test_fitness_advanced.py`

**Failing Tests:**

```
FAILED test_chunked_vs_non_chunked
FAILED test_chunked_with_large_population  
FAILED test_chunked_with_different_batch_sizes
FAILED test_multiple_fitness_functions
FAILED test_weighted_combine_function
FAILED test_fitness_with_context
FAILED test_fitness_with_different_context_values
FAILED test_fitness_with_translation
```

**Root Cause:** Tests are passing invalid key shapes to `get_fitness_function`

```python
# From error trace:
TypeError: JAX encountered invalid PRNG key data: expected key_data.shape to 
end with (2,); got shape=(2, 6)
```

**Recommendation:**

1. Fix the 8 failing tests - they're passing sequences as keys
2. Add tests for:
   - Chunked evaluation with various batch sizes
   - Multiple fitness components stacking
   - Context-dependent fitness (e.g., temperature/beta)
   - Translation integration (nucleotide → protein → fitness)

---

### 7. Runner Integration ✅ **GOOD COVERAGE**

**`runner.py`**: **95% coverage** (98 statements, 5 missed)

**Tests:** 7 tests, all passing

- ✅ Registry validation
- ✅ Config validation
- ✅ Auto-tuning integration
- ✅ Output directory creation

**Missing Coverage:** Likely edge cases in:

- Multiple sequence type handling
- Complex configuration combinations
- Error recovery paths

---

### 8. I/O System **MINOR GAPS** ⚠️

**`io.py`**: **93% coverage** (60 statements, 4 missed)

**Current Tests:** 24 tests

- ✅ 20 tests passing (git, metadata, writer, callbacks)
- ❌ **4 tests failing** in `read_lineage_data`

**Failing Tests:**

```
FAILED test_read_lineage_data_basic
FAILED test_read_lineage_data_with_arrays
FAILED test_read_lineage_data_complex_pytree  
FAILED test_pytree_roundtrip_preserves_structure
```

**Root Cause:** Data structure mismatch - tests expect `{"data": {...}}` but get `{"leaves": [...]}`

**Recommendation:**

1. Fix the 4 failing read tests - update assertions to match actual PyTree serialization format
2. Add tests for ArrayRecord edge cases

---

### 9. Initialization Factory **MODERATE ISSUES** ⚠️

**`sampling/initialization_factory.py`**: **92% coverage** (97 statements, 8 missed)

**Tests:** 22 tests total

- ✅ 16 tests passing
- ❌ **4 tests failing** (PRSMC and SMC state initialization)
- ⏭️ **2 tests skipped** (HMC/NUTS with int8)

**Failing Tests:**

```
FAILED test_initialize_parallel_replica_state
FAILED test_mcmc_initialization  
FAILED test_smc_state_initialization
FAILED test_smc_state_with_none_beta
FAILED test_prsmc_state_initialization
FAILED test_prsmc_additional_fields_values
```

**Known Issue (from AGENTS.md):**
> **Data Type Conversions (HMC/NUTS):** HMC/NUTS samplers require float32 inputs for gradient computation, but sequences are stored as int8 for memory efficiency.

**Recommendation:**

1. Address the documented dtype conversion issue
2. Fix the 6 failing initialization tests
3. Unskip and fix the 2 HMC/NUTS tests

---

### 10. **HIGH PRIORITY GAP:** Sampler Loops ❌ **MAJOR GAPS**

#### MCMC Loop

**`sampling/mcmc.py`**: **87% coverage** but **4/4 tests failing**

```
FAILED test_loop_executes
FAILED test_loop_with_io_callback
FAILED test_loop_state_progression  
FAILED test_loop_returns_empty_metrics
```

#### HMC Loop

**`sampling/hmc.py`**: **36% coverage** (42 statements, 27 missed)

- **5 tests skipped** - "Tests need to be updated for new HMC loop implementation"

#### NUTS Loop  

**`sampling/nuts.py`**: **37% coverage** (43 statements, 27 missed)

- **6/6 tests failing**

```
FAILED test_loop_executes
FAILED test_loop_with_io_callback
FAILED test_loop_state_progression
FAILED test_loop_returns_empty_metrics
FAILED test_loop_with_different_step_sizes
FAILED test_loop_ignores_mutation_fn
```

#### Gibbs Loop

**`sampling/gibbs.py`**: **48% coverage** (48 statements, 25 missed)

- Tests exist for `make_gibbs_update_fns` but not for `run_gibbs_loop`

**Why Critical:**

- These are core sampling algorithms
- All have significant implementation or test issues
- Poor coverage suggests potential bugs

**Recommendation:**

1. **Immediate:** Fix or document why MCMC/NUTS tests are failing
2. **Short-term:** Update skipped HMC tests per their TODO comment
3. **Medium-term:** Add comprehensive `run_gibbs_loop` tests
4. **Consider:** Whether these samplers should be marked as experimental/unstable

---

### 11. **CRITICAL GAP:** ESM Integration ❌ **HALF COVERED**

**`utils/esm.py`**: **47% coverage** (215 statements, 113 missed)

**Why Critical:**

- **Largest untested module** (215 statements)
- Integrates external ESM model
- Complex external dependency
- High potential for integration bugs

**Current Tests:** Basic factory and execution tests pass

**Recommendation:** Add comprehensive tests for:

- Model loading and initialization
- Batch processing
- Error handling (model unavailable, GPU memory issues)
- Different ESM model variants
- Edge cases (empty sequences, very long sequences)

---

### 12. Metrics and Annealing **MODERATE GAPS** ⚠️

**`utils/metrics.py`**: **66% coverage** (73 statements, 25 missed)
**`utils/annealing.py`**: **62% coverage** (61 statements, 23 missed)

**Current Status:**

- Basic functionality tested
- Edge cases and error paths likely untested
- Important for SMC convergence analysis

**Recommendation:** Add tests for:

- Metrics edge cases (empty populations, NaN values, inf weights)
- All annealing schedule types beyond linear
- Annealing schedule validation and error cases

---

### 13. Utilities - Mixed Status

| Module | Coverage | Status |
|--------|----------|--------|
| `utils/mutation.py` | 100% ✅ | Excellent |
| `utils/translation.py` | 100% ✅ | Excellent |
| `utils/initiate.py` | 100% ✅ | Excellent |
| `utils/blackjax_utils.py` | 100% ✅ | Excellent |
| `utils/config_unpacker.py` | 100% ✅ | Excellent |
| `utils/key_management.py` | 100% ✅ | Excellent |
| `utils/jax_utils.py` | 96% ✅ | Very Good |
| `utils/memory.py` | 89% ✅ | Good |
| `utils/constants.py` | 92% ✅ | Good |
| `utils/nk_landscape.py` | 77% ⚠️ | Acceptable |
| `utils/pmap_utils.py` | 35% ❌ | Poor |
| `utils/jax_uuid.py` | 0% ❌ | **No tests** |

---

## Priority Action Plan

### Immediate (This Sprint)

1. **Fix Failing Tests** (29 failing tests)
   - Fix `test_fitness_advanced.py` key/sequence confusion (8 tests)
   - Fix `test_io.py` PyTree structure assertions (4 tests)
   - Fix `test_initialization_factory.py` dtype issues (6 tests)
   - Fix `test_mcmc_loop.py` integration issues (4 tests)
   - Fix `test_nuts_loop.py` implementation issues (6 tests)
   - Fix `test_oed/test_experiment_integration.py` (1 test)

2. **Address Critical Zero-Coverage Modules**
   - Add `tests/models/test_protocols.py`
   - Add tests for `models/mutation.py` type definitions
   - Add tests for `models/translation.py` protocols
   - Add tests for `utils/jax_uuid.py`

3. **Document Known Issues**
   - Update AGENTS.md with current test failures
   - Add TODO comments in failing test files
   - Consider marking unstable samplers with warnings

### Short-term (Next 2 Sprints)

4. **Improve Sampler Loop Coverage**
   - Update and unskip HMC tests (5 skipped)
   - Add comprehensive Gibbs loop tests
   - Ensure MCMC/NUTS loops are stable

5. **ESM Integration Testing**
   - Add comprehensive `test_esm.py` coverage
   - Mock external model dependencies
   - Test error handling and edge cases

6. **Fitness Evaluation Improvements**
   - Increase `utils/fitness.py` coverage to >85%
   - Test chunked evaluation thoroughly
   - Test multi-component fitness stacking

### Medium-term (Next Quarter)

7. **Improve Metrics and Annealing**
   - Increase `utils/metrics.py` to >85%
   - Test all annealing schedules
   - Add property-based tests for statistical functions

8. **Memory Management Testing**
   - Increase `models/memory.py` to >85%
   - Test auto-tuning edge cases
   - Add stress tests for large populations

9. **Parallel Utilities**
   - Increase `utils/pmap_utils.py` from 35% to >85%
   - Test distributed computation scenarios

---

## Testing Philosophy Recommendations

### What's Working Well ✅

- **Parallel Replica SMC**: 100% coverage, comprehensive tests
- **Mutation utilities**: Clear, thorough testing
- **JAX utilities**: Well-structured tests with edge cases
- **Configuration unpacking**: Excellent test design

### Areas for Improvement ⚠️

1. **Test Organization**
   - Some test files have failing tests that need immediate attention
   - Consider separating integration tests from unit tests
   - Add markers for slow tests (ESM, OED)

2. **Mock Usage**
   - Heavy external dependencies (ESM) need better mocking
   - I/O operations could use more mocks for faster tests

3. **Property-Based Testing**
   - Statistical functions (metrics, annealing) would benefit from Hypothesis tests
   - Consider adding property tests for mutation invariants

4. **Integration Testing**
   - End-to-end SMC runs are tested, but need more scenarios
   - Multi-sampler comparisons would catch integration issues

5. **Performance Testing**
   - No performance regression tests
   - Consider adding benchmarks for critical paths

---

## Coverage Targets by Module Type

| Module Type | Current | Target | Priority |
|-------------|---------|--------|----------|
| Core SMC | 95%+ | 100% | High |
| Samplers (MCMC/HMC/NUTS/Gibbs) | 36-87% | 90%+ | **Critical** |
| Protocols & Types | 0% | 80%+ | **Critical** |
| Utilities | 60-100% | 90%+ | Medium |
| I/O | 93% | 95%+ | Medium |
| Runner | 95% | 95%+ | Low (maintain) |
| OED | 62-98% | 80%+ | Low |

---

## Conclusion

The proteinsmc codebase has **solid coverage (80%)** in core areas like SMC, parallel replica, and utilities, but suffers from **critical gaps** in:

1. **Protocol definitions** (0% coverage) - affects type safety
2. **Sampler loops** (36-87% coverage, many failing tests) - affects reliability
3. **ESM integration** (47% coverage) - large untested surface area

**Immediate action required:**

- Fix 29 failing tests
- Add tests for 0% coverage modules
- Stabilize or mark unstable sampler implementations

**Success would look like:**

- All tests passing (0 failures)
- >90% coverage on core modules
- <5 skipped tests (with documented reasons)
- Clear integration test suite for end-to-end workflows
