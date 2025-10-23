"""Type protocols for standardized sampler interfaces.

This module defines Protocol classes that establish canonical function signatures
for fitness evaluation, mutation operations, and other core sampler components.
All samplers should implement functions that conform to these protocols.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
  from jaxtyping import Array, Float, PRNGKeyArray


class FitnessFn(Protocol):
  """Canonical fitness function signature.

  All fitness functions across samplers should conform to this interface for consistency.
  The context parameter allows flexible passing of temperature schedules, step counts,
  and other metadata without changing the function signature.

  Args:
    key: PRNG key for stochastic fitness evaluation.
    sequence: Protein sequence(s) to evaluate. Shape (seq_len,) for single sequence
      or (pop_size, seq_len) for population.
    context: Optional metadata dict. Common keys include:
      - 'beta': Float temperature/annealing parameter.
      - 'step': Int current iteration number.
      - 'mutation_rate': Float for adaptive mutation.

  Returns:
    Fitness scores with shape matching input batching:
    - (,) for single sequence input.
    - (pop_size,) for population input.

  Example:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> key = jax.random.PRNGKey(42)
    >>> sequence = jnp.array([0, 1, 2, 3])
    >>> context = {"beta": 1.0, "step": 0}
    >>> fitness = fitness_fn(key, sequence, context)

  """

  def __call__(
    self,
    key: PRNGKeyArray,
    sequence: Array,
    context: dict[str, Array | float | int] | None = None,
  ) -> Float[Array, ...]:
    """Evaluate fitness of sequence(s)."""
    ...


class MutationFn(Protocol):
  """Canonical mutation function signature.

  All mutation functions should accept a PRNG key and sequence(s), with optional
  context for adaptive parameters.

  Args:
    key: PRNG key for stochastic mutations.
    sequence: Sequence(s) to mutate. Shape (seq_len,) or (pop_size, seq_len).
    context: Optional metadata dict. Common keys include:
      - 'mutation_rate': Float mutation probability per position.
      - 'temperature': Float for temperature-dependent mutation.
      - 'step': Int current iteration for adaptive strategies.

  Returns:
    Mutated sequence(s) with same shape as input.

  Example:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> key = jax.random.PRNGKey(42)
    >>> sequence = jnp.array([0, 1, 2, 3])
    >>> context = {"mutation_rate": 0.1}
    >>> mutated = mutation_fn(key, sequence, context)

  """

  def __call__(
    self,
    key: PRNGKeyArray,
    sequence: Array,
    context: dict[str, Array | float | int] | None = None,
  ) -> Array:
    """Apply mutations to sequence(s)."""
    ...


class AnnealingFn(Protocol):
  """Canonical annealing schedule signature.

  Annealing functions compute temperature parameters for tempered sampling algorithms.

  Args:
    step: Current iteration number (0-indexed).
    context: Optional metadata dict for complex schedules.

  Returns:
    Temperature/beta parameter for current step.

  Example:
    >>> beta = annealing_fn(step=10, context={"total_steps": 100})

  """

  def __call__(
    self,
    step: int,
    context: dict[str, Array | float | int] | None = None,
  ) -> Float:
    """Compute annealing parameter for given step."""
    ...
