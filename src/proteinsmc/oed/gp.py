"""Gaussian Process regression implementation for OED surrogate modeling."""

import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float

from proteinsmc.oed.structs import OEDDesign, OEDFeatureMode, OEDPredictedVariables


@struct.dataclass
class GPModel:
  """Gaussian Process regression model for surrogate modeling.

  Attributes:
    X_train: Training input features.
    y_train: Training target values.
    kernel_fn: Kernel function.
    noise: Observation noise variance.
    length_scale: Length scale parameter for RBF kernel.
    signal_variance: Signal variance parameter for RBF kernel.

  """

  X_train: Array
  y_train: Array
  X_mean: Array
  X_std: Array
  noise: Float = 1e-5
  length_scale: Float = 1.0
  signal_variance: Float = 1.0

  def rbf_kernel(self, x1: Array, x2: Array) -> Array:
    """Radial basis function (RBF) kernel.

    Args:
      x1: First input array.
      x2: Second input array.

    Returns:
      Kernel matrix.

    Example:
      >>> model = GPModel(X_train=jnp.array([[1.0, 2.0]]), y_train=jnp.array([0.5]))
      >>> k = model.rbf_kernel(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0]))
      >>> jnp.isclose(k, model.signal_variance)
      True

    """
    # Compute squared Euclidean distance
    sq_norm = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
    # Apply RBF kernel formula: k(x, x') = σ² * exp(-||x - x'||² / (2 * l²))
    return self.signal_variance * jnp.exp(-0.5 * sq_norm / (self.length_scale**2))

  def predict(self, x_new: Array) -> tuple[Array, Array]:
    """Make predictions at new input points.

    Args:
      x_new: New input points for prediction.

    Returns:
      Tuple of (mean, variance) predictions.

    Example:
      >>> X_train = jnp.array([[1.0, 2.0], [3.0, 4.0]])
      >>> y_train = jnp.array([0.5, 1.5])
      >>> model = GPModel(X_train=X_train, y_train=y_train)
      >>> mean, var = model.predict(jnp.array([[2.0, 3.0]]))
      >>> mean.shape == (1,)
      True
      >>> var.shape == (1,)
      True

    """
    # Normalize new input
    x_new_norm = (x_new - self.X_mean) / (self.X_std + 1e-8)
    # X_train is already normalized

    # Compute kernel matrices
    k = self.rbf_kernel(self.X_train, self.X_train)
    k_s = self.rbf_kernel(self.X_train, x_new_norm)
    k_ss = self.rbf_kernel(x_new_norm, x_new_norm)

    # Add noise to diagonal
    k_noise = k + self.noise * jnp.eye(k.shape[0])

    # Compute mean: K_s^T * K^-1 * y
    # Use a small jitter for stability if Cholesky fails
    jitter = 1e-6
    k_noise_stable = k_noise + jitter * jnp.eye(k_noise.shape[0])

    l_mat = jnp.linalg.cholesky(k_noise_stable)
    alpha = jnp.linalg.solve(l_mat.T, jnp.linalg.solve(l_mat, self.y_train))
    mu = jnp.matmul(k_s.T, alpha)

    # Compute variance: K_ss - K_s^T * K^-1 * K_s
    v = jnp.linalg.solve(l_mat, k_s)
    var = jnp.diag(k_ss - jnp.matmul(v.T, v))

    # Ensure variance is non-negative
    var = jnp.clip(var, a_min=1e-9)

    return mu, var


def fit_gp_model(
  x: Array,
  y: Array,
  noise: Float = 1e-5,
  length_scale: Float = 1.0,
  signal_variance: Float = 1.0,
) -> dict[str, GPModel]:
  """Fit a GP model for each output dimension.

  Args:
    x: Input features, shape (n_samples, n_features).
    y: Output values, shape (n_samples, n_outputs).
    noise: Observation noise variance.
    length_scale: Length scale for RBF kernel.
    signal_variance: Signal variance for RBF kernel.

  Returns:
    Dictionary of GP models, one per output dimension.

  Example:
    >>> X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    >>> Y = jnp.array([[0.5, 0.6], [1.5, 1.6]])
    >>> models = fit_gp_model(X, Y)
    >>> len(models) == Y.shape[1]
    True

  """
  output_dims = y.shape[1]
  models = {}

  # Calculate normalization parameters
  x_mean = jnp.mean(x, axis=0)
  x_std = jnp.std(x, axis=0) + 1e-8
  x_norm = (x - x_mean) / x_std

  # Fit a separate GP for each output dimension
  for i in range(output_dims):
    models[f"dim_{i}"] = GPModel(
      X_train=x_norm,
      y_train=y[:, i],
      X_mean=x_mean,
      X_std=x_std,
      noise=noise,
      length_scale=length_scale,
      signal_variance=signal_variance,
    )

  return models


def predict_with_gp_models(
  models: dict[str, GPModel], x_new: Array
) -> tuple[dict[str, Array], dict[str, Array]]:
  """Make predictions using GP models for all output dimensions.

  Args:
    models: Dictionary of GP models.
    x_new: New input points for prediction.

  Returns:
    Tuple of dictionaries containing means and variances.

  Example:
    >>> X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    >>> Y = jnp.array([[0.5, 0.6], [1.5, 1.6]])
    >>> models = fit_gp_model(X, Y)
    >>> means, vars = predict_with_gp_models(models, jnp.array([[2.0, 3.0]]))
    >>> list(means.keys()) == list(models.keys())
    True

  """
  means = {}
  variances = {}

  for name, model in models.items():
    mu, var = model.predict(x_new)
    means[name] = mu
    variances[name] = var

  return means, variances


def design_to_features(design: OEDDesign, mode: OEDFeatureMode = OEDFeatureMode.ALL) -> Array:
  """Convert OEDDesign to feature array for GP input.

  Args:
    design: OED design parameters.
    mode: Feature selection mode.

  Returns:
    Feature array for GP input.

  Example:
    >>> from proteinsmc.oed.structs import OEDDesign, OEDFeatureMode
    >>> design = OEDDesign(N=20, K=3, q=4, population_size=100,
    ...                    n_generations=50, mutation_rate=0.01,
    ...                    diversification_ratio=0.1)
    >>> features = design_to_features(design, mode=OEDFeatureMode.ALL)
    >>> features.shape == (1, 8)
    True
    >>> features_eff = design_to_features(design, mode=OEDFeatureMode.EFFECTIVE_ONLY)
    >>> features_eff.shape == (1, 5)
    True

  """
  features = [
    design.N,
    design.K,
    design.q,
    design.population_size,
    design.mutation_rate,
    design.diversification_ratio,
    design.branch_length,
  ]
  # Effective mutation rate (population_size * branch_length * mutation_rate)
  eff_mut_rate = design.population_size * design.branch_length * design.mutation_rate
  features.append(eff_mut_rate)

  if mode == OEDFeatureMode.EFFECTIVE_ONLY:
    # Exclude subcomponents: population_size, mutation_rate, branch_length
    # Indices: N(0), K(1), q(2), pop(3), mut(4), div(5), branch(6), eff(7)
    # Remaining: N, K, q, div, eff
    indices = [0, 1, 2, 5, 7]
    features = [features[i] for i in indices]

  return jnp.array([features])


def features_to_predicted_variables(
  mean_dict: dict[str, Array], var_dict: dict[str, Array]
) -> OEDPredictedVariables:
  """Convert GP prediction dictionaries to OEDPredictedVariables.

  Args:
    mean_dict: Dictionary of mean predictions.
    var_dict: Dictionary of variance predictions.

  Returns:
    OEDPredictedVariables with predictions.

  Example:
    >>> mean_dict = {"dim_0": jnp.array([0.5]), "dim_1": jnp.array([0.6]),
    ...              "dim_2": jnp.array([0.7]), "dim_3": jnp.array([0.8]),
    ...              "dim_4": jnp.array([0.9])}
    >>> var_dict = {"dim_0": jnp.array([0.1]), "dim_1": jnp.array([0.1]),
    ...             "dim_2": jnp.array([0.1]), "dim_3": jnp.array([0.1]),
    ...             "dim_4": jnp.array([0.1])}
    >>> vars = features_to_predicted_variables(mean_dict, var_dict)
    >>> jnp.isclose(vars.information_gain, 0.5)
    True

  """
  del var_dict
  return OEDPredictedVariables(
    information_gain=mean_dict["dim_0"][0],
    barrier_crossing_frequency=mean_dict["dim_1"][0],
    final_sequence_entropy=mean_dict["dim_2"][0],
    jsd_from_original_population=mean_dict["dim_3"][0],
    geometric_fitness_mean=mean_dict["dim_4"][0],
  )
