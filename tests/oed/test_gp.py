import jax.numpy as jnp
from proteinsmc.oed.gp import fit_gp_model, predict_with_gp_models, design_to_features
from proteinsmc.oed.structs import OEDDesign


def test_design_to_features_shape():
    design = OEDDesign(N=10, K=2, q=4, population_size=20, n_generations=5, mutation_rate=0.01, diversification_ratio=0.1)
    features = design_to_features(design)
    assert features.shape == (1, 6)


def test_fit_and_predict():
    X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    Y = jnp.array([[0.5, 0.6], [1.5, 1.6]])
    models = fit_gp_model(X, Y)
    means, vars = predict_with_gp_models(models, jnp.array([[2.0, 3.0]]))
    assert set(means.keys()) == set(models.keys())
    for v in means.values():
        assert v.shape[0] == 1

