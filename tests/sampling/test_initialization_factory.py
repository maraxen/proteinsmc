"""Tests for initialization factory module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from blackjax.smc.base import SMCState as BaseSMCState

from proteinsmc.models.fitness import StackedFitness
from proteinsmc.models.sampler_base import SamplerState
from proteinsmc.models.smc import SMCAlgorithmType
from proteinsmc.sampling.initialization_factory import (
  _initialize_blackjax_smc_state,
  _initialize_prsmc_state,
  _initialize_single_state,
  _initialize_smc_state,
  initialize_sampler_state,
)


@pytest.fixture
def mock_fitness_fn():
  """Create a mock fitness function for testing.

  Returns:
      A callable that returns stacked fitness values.

  Example:
      >>> fitness_fn = mock_fitness_fn()
      >>> result = fitness_fn(key, sequence, beta)

  """

  def fitness_fn(key, seq, beta):  # noqa: ARG001
    """Mock fitness function.

    Args:
        seq: Input sequence.
        key: PRNG key (unused).
        beta: Temperature parameter (unused).

    Returns:
        StackedFitness: Mock fitness values.

    """
    if seq.ndim > 1:
        return jnp.tile(jnp.array([1.0, 0.5, 0.3], dtype=jnp.float32), (seq.shape[0], 1))
    return jnp.array([1.0, 0.5, 0.3], dtype=jnp.float32)

  return fitness_fn


class TestInitializeSamplerState:
  """Test the main initialize_sampler_state function."""

  def test_initialize_mcmc_state(
    self,
    rng_key,
    mock_fitness_fn,
    sample_protein_sequence,
  ) -> None:
    """Test initialization of MCMC sampler state.

    Args:
        rng_key: PRNG key fixture.
        mock_fitness_fn: Mock fitness function fixture.
        sample_protein_sequence: Sample protein sequence fixture.

    Returns:
        None

    Raises:
        AssertionError: If state initialization fails.

    Example:
        >>> test_initialize_mcmc_state(rng_key, mock_fitness_fn, sample_protein_sequence)

    """
    state = initialize_sampler_state(
      sampler_type="MCMC",
      sequence_type="protein",
      seed_sequence=sample_protein_sequence,
      mutation_rate=0.1,
      population_size=1,
      algorithm=None,
      smc_algo_kwargs=None,
      n_islands=None,
      population_size_per_island=None,
      island_betas=None,
      diversification_ratio=None,
      key=rng_key,
      beta=None,
      fitness_fn=mock_fitness_fn,
    )

    assert isinstance(state, SamplerState)
    assert state.sequence.shape == sample_protein_sequence.shape
    assert state.step == 0
    assert state.blackjax_state is not None

  @pytest.mark.skip(reason="HMC initialization test is currently skipped. We will implement later once we have integrated managing sequence representations as both integer and one-hot encoding.")
  def test_initialize_hmc_state(
    self,
    rng_key,
    mock_fitness_fn,
    sample_protein_sequence,
  ) -> None:
    """Test initialization of HMC sampler state.

    Args:
        rng_key: PRNG key fixture.
        mock_fitness_fn: Mock fitness function fixture.
        sample_protein_sequence: Sample protein sequence fixture.

    Returns:
        None

    Raises:
        AssertionError: If state initialization fails.

    Example:
        >>> test_initialize_hmc_state(rng_key, mock_fitness_fn, sample_protein_sequence)

    """
    state = initialize_sampler_state(
      sampler_type="HMC",
      sequence_type="protein",
      seed_sequence=sample_protein_sequence,
      mutation_rate=0.1,
      population_size=1,
      algorithm=None,
      smc_algo_kwargs=None,
      n_islands=None,
      population_size_per_island=None,
      island_betas=None,
      diversification_ratio=None,
      key=rng_key,
      beta=None,
      fitness_fn=mock_fitness_fn,
    )

    assert isinstance(state, SamplerState)
    assert state.sequence.shape == sample_protein_sequence.shape
    assert state.step == 0

  @pytest.mark.skip(reason="HMC initialization test is currently skipped. We will implement later once we have integrated managing sequence representations as both integer and one-hot encoding.")
  def test_initialize_nuts_state(
    self,
    rng_key,
    mock_fitness_fn,
    sample_protein_sequence,
  ) -> None:
    """Test initialization of NUTS sampler state.

    Args:
        rng_key: PRNG key fixture.
        mock_fitness_fn: Mock fitness function fixture.
        sample_protein_sequence: Sample protein sequence fixture.

    Returns:
        None

    Raises:
        AssertionError: If state initialization fails.

    Example:
        >>> test_initialize_nuts_state(rng_key, mock_fitness_fn, sample_protein_sequence)

    """
    state = initialize_sampler_state(
      sampler_type="NUTS",
      sequence_type="protein",
      seed_sequence=sample_protein_sequence,
      mutation_rate=0.1,
      population_size=1,
      algorithm=None,
      smc_algo_kwargs=None,
      n_islands=None,
      population_size_per_island=None,
      island_betas=None,
      diversification_ratio=None,
      key=rng_key,
      beta=None,
      fitness_fn=mock_fitness_fn,
    )

    assert isinstance(state, SamplerState)
    assert state.sequence.shape == sample_protein_sequence.shape
    assert state.step == 0

  def test_initialize_smc_state_basic(
    self,
    rng_key,
    mock_fitness_fn,
    sample_protein_sequence,
  ) -> None:
    """Test initialization of basic SMC sampler state.

    Args:
        rng_key: PRNG key fixture.
        mock_fitness_fn: Mock fitness function fixture.
        sample_protein_sequence: Sample protein sequence fixture.

    Returns:
        None

    Raises:
        AssertionError: If state initialization fails.

    Example:
        >>> test_initialize_smc_state_basic(rng_key, mock_fitness_fn, sample_protein_sequence)

    """
    state = initialize_sampler_state(
      sampler_type="SMC",
      sequence_type="protein",
      seed_sequence=sample_protein_sequence,
      mutation_rate=0.1,
      population_size=10,
      algorithm="BaseSMC",
      smc_algo_kwargs={},
      n_islands=None,
      population_size_per_island=None,
      island_betas=None,
      diversification_ratio=None,
      key=rng_key,
      beta=1.0,
        fitness_fn=mock_fitness_fn,
    )

    assert isinstance(state, SamplerState)
    assert state.sequence.shape == (10, *sample_protein_sequence.shape)
    assert state.step == 0
    assert "beta" in state.additional_fields

  def test_initialize_smc_state_tempered(
    self,
    rng_key,
    mock_fitness_fn,
    sample_protein_sequence,
  ) -> None:
    """Test initialization of tempered SMC sampler state.

    Args:
        rng_key: PRNG key fixture.
        mock_fitness_fn: Mock fitness function fixture.
        sample_protein_sequence: Sample protein sequence fixture.

    Returns:
        None

    Raises:
        AssertionError: If state initialization fails.

    Example:
        >>> test_initialize_smc_state_tempered(rng_key, mock_fitness_fn, sample_protein_sequence)

    """
    state = initialize_sampler_state(
      sampler_type="SMC",
      sequence_type="protein",
      seed_sequence=sample_protein_sequence,
      mutation_rate=0.1,
      population_size=10,
      algorithm="TemperedSMC",
      smc_algo_kwargs={},
      n_islands=None,
      population_size_per_island=None,
      island_betas=None,
      diversification_ratio=None,
      key=rng_key,
      beta=0.5,
      fitness_fn=mock_fitness_fn,
    )

    assert isinstance(state, SamplerState)
    assert state.additional_fields["beta"] == 0.5

  def test_initialize_parallel_replica_state(
    self,
    rng_key,
    mock_fitness_fn,
    sample_protein_sequence,
  ) -> None:
    """Test initialization of Parallel Replica SMC sampler state.

    Args:
        rng_key: PRNG key fixture.
        mock_fitness_fn: Mock fitness function fixture.
        sample_protein_sequence: Sample protein sequence fixture.

    Returns:
        None

    Raises:
        AssertionError: If state initialization fails.

    Example:
        >>> test_initialize_parallel_replica_state(rng_key, mock_fitness_fn, 
        ...     sample_protein_sequence)

    """
    n_islands = 4
    pop_per_island = 10
    island_betas = jnp.array([0.1, 0.3, 0.5, 1.0], dtype=jnp.float32)

    state = initialize_sampler_state(
      sampler_type="ParallelReplica",
      sequence_type="protein",
      seed_sequence=sample_protein_sequence,
      mutation_rate=0.1,
      population_size=pop_per_island,  # Used to generate initial population
      algorithm=None,
      smc_algo_kwargs=None,
      n_islands=n_islands,
      population_size_per_island=pop_per_island,
      island_betas=island_betas,
      diversification_ratio=0.2,
      key=rng_key,
      beta=None,
      fitness_fn=mock_fitness_fn,
    )

    assert isinstance(state, SamplerState)
    assert state.sequence.shape == (n_islands, pop_per_island, *sample_protein_sequence.shape)
    assert state.step.shape == (n_islands,)
    assert "beta" in state.additional_fields
    assert "mean_fitness" in state.additional_fields
    assert "max_fitness" in state.additional_fields

  def test_unsupported_sampler_type_raises_error(
    self,
    rng_key,
    mock_fitness_fn,
    sample_protein_sequence,
  ) -> None:
    """Test that unsupported sampler type raises ValueError.

    Args:
        rng_key: PRNG key fixture.
        mock_fitness_fn: Mock fitness function fixture.
        sample_protein_sequence: Sample protein sequence fixture.

    Returns:
        None

    Raises:
        ValueError: If sampler type is not supported.

    Example:
        >>> test_unsupported_sampler_type_raises_error(rng_key, mock_fitness_fn, 
        ...     sample_protein_sequence)

    """
    with pytest.raises(ValueError, match="Unsupported sampler type"):
      initialize_sampler_state(
        sampler_type="InvalidSampler",
        sequence_type="protein",
        seed_sequence=sample_protein_sequence,
        mutation_rate=0.1,
        population_size=10,
        algorithm=None,
        smc_algo_kwargs=None,
        n_islands=None,
        population_size_per_island=None,
        island_betas=None,
        diversification_ratio=None,
        key=rng_key,
        beta=None,
        fitness_fn=mock_fitness_fn,
      )


class TestInitializeSingleState:
  """Test the _initialize_single_state helper function."""

  def test_mcmc_initialization(
    self,
    rng_key,
    mock_fitness_fn,
    sample_population_proteins,
  ) -> None:
    """Test MCMC state initialization.

    Args:
        rng_key: PRNG key fixture.
        mock_fitness_fn: Mock fitness function fixture.
        sample_population_proteins: Sample population fixture.

    Returns:
        None

    Raises:
        AssertionError: If initialization fails.

    Example:
        >>> test_mcmc_initialization(rng_key, mock_fitness_fn, sample_population_proteins)

    """
    state = _initialize_single_state(
      sampler_type="MCMC",
      initial_population=sample_population_proteins,
      fitness_fn=mock_fitness_fn,
      key=rng_key,
    )

    assert isinstance(state, SamplerState)
    assert state.sequence.shape == sample_population_proteins[0].shape
  @pytest.mark.skip(reason="HMC initialization test is currently skipped. We will implement later once we have integrated managing sequence representations as both integer and one-hot encoding.")
  def test_hmc_initialization(
    self,
    rng_key,
    mock_fitness_fn,
    sample_population_proteins,
  ) -> None:
    """Test HMC state initialization.

    Args:
        rng_key: PRNG key fixture.
        mock_fitness_fn: Mock fitness function fixture.
        sample_population_proteins: Sample population fixture.

    Returns:
        None

    Raises:
        AssertionError: If initialization fails.

    Example:
        >>> test_hmc_initialization(rng_key, mock_fitness_fn, sample_population_proteins)

    """
    state = _initialize_single_state(
      sampler_type="HMC",
      initial_population=sample_population_proteins,
      fitness_fn=mock_fitness_fn,
      key=rng_key,
    )

    assert isinstance(state, SamplerState)
    assert state.blackjax_state is not None

  @pytest.mark.skip(reason="HMC initialization test is currently skipped. We will implement later once we have integrated managing sequence representations as both integer and one-hot encoding.")
  def test_nuts_initialization(
    self,
    rng_key,
    mock_fitness_fn,
    sample_population_proteins,
  ) -> None:
    """Test NUTS state initialization.

    Args:
        rng_key: PRNG key fixture.
        mock_fitness_fn: Mock fitness function fixture.
        sample_population_proteins: Sample population fixture.

    Returns:
        None

    Raises:
        AssertionError: If initialization fails.

    Example:
        >>> test_nuts_initialization(rng_key, mock_fitness_fn, sample_population_proteins)

    """
    state = _initialize_single_state(
      sampler_type="NUTS",
      initial_population=sample_population_proteins,
      fitness_fn=mock_fitness_fn,
      key=rng_key,
    )

    assert isinstance(state, SamplerState)
    assert state.blackjax_state is not None

  def test_invalid_sampler_type_raises_error(
    self,
    rng_key,
    mock_fitness_fn,
    sample_population_proteins,
  ) -> None:
    """Test that invalid sampler type raises ValueError.

    Args:
        rng_key: PRNG key fixture.
        mock_fitness_fn: Mock fitness function fixture.
        sample_population_proteins: Sample population fixture.

    Returns:
        None

    Raises:
        ValueError: If sampler type is invalid.

    Example:
        >>> test_invalid_sampler_type_raises_error(rng_key, mock_fitness_fn, 
        ...     sample_population_proteins)

    """
    with pytest.raises(ValueError, match="Unsupported single-particle sampler type"):
      _initialize_single_state(
        sampler_type="InvalidType",
        initial_population=sample_population_proteins,
        fitness_fn=mock_fitness_fn,
        key=rng_key,
      )


class TestInitializeBlackjaxSMCState:
  """Test the _initialize_blackjax_smc_state helper function."""

  def test_base_smc_algorithm(self, rng_key, sample_population_proteins) -> None:
    """Test base SMC algorithm initialization.

    Args:
        rng_key: PRNG key fixture.
        sample_population_proteins: Sample population fixture.

    Returns:
        None

    Raises:
        AssertionError: If initialization fails.

    Example:
        >>> test_base_smc_algorithm(rng_key, sample_population_proteins)

    """
    state = _initialize_blackjax_smc_state(
      key=rng_key,
      algorithm="BaseSMC",
      initial_population=sample_population_proteins,
    )

    assert isinstance(state, BaseSMCState)
    assert state.particles.shape == sample_population_proteins.shape # type: ignore

  def test_tempered_smc_algorithm(self, rng_key, sample_population_proteins) -> None:
    """Test tempered SMC algorithm initialization.

    Args:
        rng_key: PRNG key fixture.
        sample_population_proteins: Sample population fixture.

    Returns:
        None

    Raises:
        AssertionError: If initialization fails.

    Example:
        >>> test_tempered_smc_algorithm(rng_key, sample_population_proteins)

    """
    state = _initialize_blackjax_smc_state(
      key=rng_key,
      algorithm="TemperedSMC",
      initial_population=sample_population_proteins,
    )

    assert state is not None
    assert hasattr(state, "particles")

  def test_adaptive_tempered_smc_algorithm(
    self,
    rng_key,
    sample_population_proteins,
  ) -> None:
    """Test adaptive tempered SMC algorithm initialization.

    Args:
        rng_key: PRNG key fixture.
        sample_population_proteins: Sample population fixture.

    Returns:
        None

    Raises:
        AssertionError: If initialization fails.

    Example:
        >>> test_adaptive_tempered_smc_algorithm(rng_key, sample_population_proteins)

    """
    state = _initialize_blackjax_smc_state(
      key=rng_key,
      algorithm="AdaptiveTemperedSMC",
      initial_population=sample_population_proteins,
    )

    assert state is not None
    assert hasattr(state, "particles")

  def test_partial_posteriors_smc_algorithm(
    self,
    rng_key,
    sample_population_proteins,
  ) -> None:
    """Test partial posteriors SMC algorithm initialization.

    Args:
        rng_key: PRNG key fixture.
        sample_population_proteins: Sample population fixture.

    Returns:
        None

    Raises:
        AssertionError: If initialization fails.

    Example:
        >>> test_partial_posteriors_smc_algorithm(rng_key, sample_population_proteins)

    """
    state = _initialize_blackjax_smc_state(
      key=rng_key,
      algorithm="PartialPosteriors",
      initial_population=sample_population_proteins,
      smc_algo_kwargs={"num_datapoints": 10},
    )

    assert state is not None
    assert hasattr(state, "particles")

  def test_inner_mcmc_raises_not_implemented(
    self,
    rng_key,
    sample_population_proteins,
  ) -> None:
    """Test that inner MCMC algorithm raises NotImplementedError.

    Args:
        rng_key: PRNG key fixture.
        sample_population_proteins: Sample population fixture.

    Returns:
        None

    Raises:
        NotImplementedError: If algorithm is not implemented.

    Example:
        >>> test_inner_mcmc_raises_not_implemented(rng_key, sample_population_proteins)

    """
    with pytest.raises(NotImplementedError, match="Inner MCMC SMC algorithm"):
      _initialize_blackjax_smc_state(
        key=rng_key,
        algorithm="InnerMCMC",
        initial_population=sample_population_proteins,
      )

  def test_pretuning_raises_not_implemented(
    self,
    rng_key,
    sample_population_proteins,
  ) -> None:
    """Test that pretuning algorithm raises NotImplementedError.

    Args:
        rng_key: PRNG key fixture.
        sample_population_proteins: Sample population fixture.

    Returns:
        None

    Raises:
        NotImplementedError: If algorithm is not implemented.

    Example:
        >>> test_pretuning_raises_not_implemented(rng_key, sample_population_proteins)

    """
    with pytest.raises(NotImplementedError, match="Pretuning SMC algorithm"):
      _initialize_blackjax_smc_state(
        key=rng_key,
        algorithm="PretuningSMC",
        initial_population=sample_population_proteins,
      )

  def test_custom_smc_without_init_fn_raises_error(
    self,
    rng_key,
    sample_population_proteins,
  ) -> None:
    """Test that custom SMC without init_fn raises ValueError.

    Args:
        rng_key: PRNG key fixture.
        sample_population_proteins: Sample population fixture.

    Returns:
        None

    Raises:
        ValueError: If custom init function is missing.

    Example:
        >>> test_custom_smc_without_init_fn_raises_error(rng_key, sample_population_proteins)

    """
    with pytest.raises(ValueError, match="custom_init_fn"):
      _initialize_blackjax_smc_state(
        key=rng_key,
        algorithm="CustomSMC",
        initial_population=sample_population_proteins,
        smc_algo_kwargs={},
      )

  def test_custom_smc_with_init_fn(self, rng_key, sample_population_proteins) -> None:
    """Test custom SMC algorithm with provided init function.

    Args:
        rng_key: PRNG key fixture.
        sample_population_proteins: Sample population fixture.

    Returns:
        None

    Raises:
        AssertionError: If initialization fails.

    Example:
        >>> test_custom_smc_with_init_fn(rng_key, sample_population_proteins)

    """

    def custom_init_fn(particles, key):  # noqa: ARG001
      """Custom initialization function.

      Args:
          particles: Initial particles.
          key: PRNG key.

      Returns:
          BaseSMCState: Custom SMC state.

      """
      return BaseSMCState(
        particles=particles,
        weights=jnp.ones(particles.shape[0]) / particles.shape[0],
        update_parameters=jnp.array(0.0),
      )

    state = _initialize_blackjax_smc_state(
      key=rng_key,
      algorithm="CustomSMC",
      initial_population=sample_population_proteins,
      smc_algo_kwargs={"custom_init_fn": custom_init_fn},
    )

    assert isinstance(state, BaseSMCState)


class TestInitializeSMCState:
  """Test the _initialize_smc_state helper function."""

  def test_smc_state_initialization(
    self,
    rng_key,
    mock_fitness_fn,
    sample_population_proteins,
  ) -> None:
    """Test SMC state initialization with all parameters.

    Args:
        rng_key: PRNG key fixture.
        mock_fitness_fn: Mock fitness function fixture.
        sample_population_proteins: Sample population fixture.

    Returns:
        None

    Raises:
        AssertionError: If initialization fails.

    Example:
        >>> test_smc_state_initialization(rng_key, mock_fitness_fn, 
        ...     sample_population_proteins)

    """
    state = _initialize_smc_state(
      initial_population=sample_population_proteins,
      beta=0.5,
      mutation_rate=0.1,
      algorithm="BaseSMC",
      smc_algo_kwargs={},
      key=rng_key,
    )

    assert isinstance(state, SamplerState)
    assert state.sequence.shape == sample_population_proteins.shape
    assert "beta" in state.additional_fields
    assert state.additional_fields["beta"] == 0.5

  def test_smc_state_with_none_beta(
    self,
    rng_key,
    mock_fitness_fn,
    sample_population_proteins,
  ) -> None:
    """Test SMC state initialization with None beta value.

    Args:
        rng_key: PRNG key fixture.
        mock_fitness_fn: Mock fitness function fixture.
        sample_population_proteins: Sample population fixture.

    Returns:
        None

    Raises:
        AssertionError: If beta is not set to default 1.0.

    Example:
        >>> test_smc_state_with_none_beta(rng_key, mock_fitness_fn, 
        ...     sample_population_proteins)

    """
    state = _initialize_smc_state(
      initial_population=sample_population_proteins,
      beta=None,
      mutation_rate=0.1,
      algorithm="BaseSMC",
      smc_algo_kwargs={},
      key=rng_key,
    )

    assert state.additional_fields["beta"] == 1.0


class TestInitializePRSMCState:
  """Test the _initialize_prsmc_state helper function."""

  def test_prsmc_state_initialization(
    self,
    rng_key,
    mock_fitness_fn,
    sample_population_proteins,
  ) -> None:
    """Test Parallel Replica SMC state initialization.

    Args:
        rng_key: PRNG key fixture.
        mock_fitness_fn: Mock fitness function fixture.
        sample_population_proteins: Sample population fixture.

    Returns:
        None

    Raises:
        AssertionError: If initialization fails.

    Example:
        >>> test_prsmc_state_initialization(rng_key, mock_fitness_fn, 
        ...     sample_population_proteins)

    """
    n_islands = 4
    pop_per_island = 2
    island_betas = jnp.array([0.1, 0.3, 0.5, 1.0], dtype=jnp.float32)

    # Create initial populations with proper shape
    initial_pops = jnp.tile(
      sample_population_proteins[:pop_per_island],
      (n_islands, 1, 1),
    )

    state = _initialize_prsmc_state(
      initial_populations=initial_pops,
      n_islands=n_islands,
      population_size_per_island=pop_per_island,
      island_betas=island_betas,
      mutation_rate=0.1,
      key=rng_key,
      fitness_fn=mock_fitness_fn,
    )

    assert isinstance(state, SamplerState)
    assert state.sequence.shape == (n_islands, pop_per_island, 4)
    assert state.key.shape == (n_islands, 2)
    assert state.step.shape == (n_islands,)
    assert "beta" in state.additional_fields
    assert "mean_fitness" in state.additional_fields
    assert "max_fitness" in state.additional_fields
    assert "ess" in state.additional_fields
    assert "logZ_estimate" in state.additional_fields

  def test_prsmc_additional_fields_values(
    self,
    rng_key,
    mock_fitness_fn,
    sample_population_proteins,
  ) -> None:
    """Test that PRSMC additional fields have correct initial values.

    Args:
        rng_key: PRNG key fixture.
        mock_fitness_fn: Mock fitness function fixture.
        sample_population_proteins: Sample population fixture.

    Returns:
        None

    Raises:
        AssertionError: If field values are incorrect.

    Example:
        >>> test_prsmc_additional_fields_values(rng_key, mock_fitness_fn, 
        ...     sample_population_proteins)

    """
    n_islands = 3
    pop_per_island = 2
    island_betas = jnp.array([0.2, 0.6, 1.0], dtype=jnp.float32)

    initial_pops = jnp.tile(
      sample_population_proteins[:pop_per_island],
      (n_islands, 1, 1),
    )

    state = _initialize_prsmc_state(
      initial_populations=initial_pops,
      n_islands=n_islands,
      population_size_per_island=pop_per_island,
      island_betas=island_betas,
      mutation_rate=0.1,
      key=rng_key,
      fitness_fn=mock_fitness_fn,
    )

    # Check beta values match input
    assert jnp.allclose(state.additional_fields["beta"], island_betas)

    # Check ESS and logZ are initialized to zero
    assert jnp.allclose(state.additional_fields["ess"], jnp.zeros(n_islands))
    assert jnp.allclose(state.additional_fields["logZ_estimate"], jnp.zeros(n_islands))

    # Check mean and max fitness have correct shapes
    assert state.additional_fields["mean_fitness"].shape == (n_islands,)
    assert state.additional_fields["max_fitness"].shape == (n_islands,)

class TestBugFixes:
  """Tests for specific bug fixes."""

  def test_initialize_single_state_fitness_fn_arguments(
    self,
    rng_key,
    sample_protein_sequence,
  ) -> None:
    """Test that fitness_fn is called with correct argument order in _initialize_single_state."""

    # We define a strict fitness function that checks argument types
    def strict_fitness_fn(key, seq, context):
      # key should be a PRNGKey (shape (2,), uint32)
      # seq should be the sequence (shape (L,), int8/int32)

      # Check if key looks like a PRNG key
      # If swapped, key will be sequence array
      if key.shape != (2,) or key.dtype != jnp.uint32:
         # If argument order is swapped, 'key' variable will hold 'seq'
         # 'seq' variable will hold 'key'
         msg = f"Expected PRNG key as first argument, got shape {key.shape}, dtype {key.dtype}"
         raise TypeError(msg)

      return jnp.array([1.0], dtype=jnp.float32)

    # Trigger _initialize_single_state via "MCMC" sampler type
    # This should call fitness_fn inside logdensity_fn
    initialize_sampler_state(
      sampler_type="MCMC",
      sequence_type="protein",
      seed_sequence=sample_protein_sequence,
      mutation_rate=0.1,
      population_size=1,
      algorithm=None,
      smc_algo_kwargs=None,
      n_islands=None,
      population_size_per_island=None,
      island_betas=None,
      diversification_ratio=None,
      key=rng_key,
      beta=None,
      fitness_fn=strict_fitness_fn,
    )

  def test_initialize_parallel_replica_string_format(
    self,
    rng_key,
    sample_protein_sequence,
  ) -> None:
    """Test that 'PARALLEL_REPLICA' string is accepted."""

    # Mock fitness fn that handles batching
    def mock_fitness_fn(key, seq, context):
      if seq.ndim == 1:
        return jnp.array([1.0], dtype=jnp.float32)
      else:
        # Batch case
        pop_size = seq.shape[0]
        # Return (pop_size, 1) so it has 'n_fitness' dimension
        return jnp.ones((pop_size, 1), dtype=jnp.float32)

    n_islands = 2
    initialize_sampler_state(
      sampler_type="PARALLEL_REPLICA",
      sequence_type="protein",
      seed_sequence=sample_protein_sequence,
      mutation_rate=0.1,
      population_size=10,
      algorithm=None,
      smc_algo_kwargs=None,
      n_islands=n_islands,
      population_size_per_island=5,
      island_betas=jnp.array([1.0, 1.0]),
      diversification_ratio=0.1,
      key=rng_key,
      beta=None,
      fitness_fn=mock_fitness_fn,
    )
