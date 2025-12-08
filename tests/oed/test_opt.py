import jax
import jax.numpy as jnp
from proteinsmc.oed.opt import perturb_design, calculate_fim_determinant
from proteinsmc.oed.run import train_surrogate_model
from proteinsmc.oed.structs import OEDDesign, OEDPredictedVariables


def make_dummy_history():
    design = OEDDesign(N=6, K=1, q=2, population_size=8, n_generations=3, mutation_rate=0.1, diversification_ratio=0.0)
    pred = OEDPredictedVariables(
        information_gain=jnp.array(0.5),
        barrier_crossing_frequency=jnp.array(0.1),
        final_sequence_entropy=jnp.array(0.2),
        jsd_from_original_population=jnp.array(0.3),
        geometric_fitness_mean=jnp.array(0.4),
    )
    return [(design, pred)]


def test_perturb_design():
    design = OEDDesign(N=6, K=1, q=2, population_size=8, n_generations=3, mutation_rate=0.1, diversification_ratio=0.0)
    p = perturb_design(design, "mutation_rate", 0.01)
    assert isinstance(p, OEDDesign)


def test_train_surrogate_and_fim():
    history = make_dummy_history()
    surrogate = train_surrogate_model(history)
    base = history[0][0]
    pred = surrogate(base)
    assert hasattr(pred, "information_gain")
    key = jax.random.PRNGKey(0)
    fim = calculate_fim_determinant(base, surrogate, key)
    assert isinstance(fim, jax.numpy.ndarray) or hasattr(fim, "shape")

