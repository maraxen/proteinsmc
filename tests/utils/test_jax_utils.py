import jax
import jax.numpy as jnp
import chex
from proteinsmc.utils.jax_utils import chunked_map

def simple_func(x):
    return x * 2

def test_chunked_map_basic():
    """Tests basic functionality of chunked_map and compares with jax.vmap."""
    data = jnp.arange(10)
    chunk_size = 3

    expected = jax.vmap(simple_func)(data)
    actual = chunked_map(simple_func, data, chunk_size)

    chex.assert_trees_all_close(actual, expected)

def test_chunked_map_uneven_chunks():
    """Tests chunked_map with a data size that is not a multiple of the chunk size."""
    data = jnp.arange(13)
    chunk_size = 5

    expected = jax.vmap(simple_func)(data)
    actual = chunked_map(simple_func, data, chunk_size)

    chex.assert_trees_all_close(actual, expected)

def test_chunked_map_empty_input():
    """Tests chunked_map with empty input data."""
    data = jnp.array([])
    chunk_size = 5

    expected = jax.vmap(simple_func)(data)
    actual = chunked_map(simple_func, data, chunk_size)

    chex.assert_trees_all_close(actual, expected)

def test_chunked_map_pytree_dict_input():
    """Tests chunked_map with a PyTree (dict) as input."""
    data = {'a': jnp.arange(10), 'b': jnp.arange(10, 20)}
    chunk_size = 4

    def pytree_func(tree):
        return tree['a'] + tree['b']

    expected = jax.vmap(pytree_func)(data)
    actual = chunked_map(pytree_func, data, chunk_size)

    chex.assert_trees_all_close(actual, expected)

def test_chunked_map_pytree_tuple_input():
    """Tests chunked_map with a PyTree (tuple) as input."""
    data = (jnp.arange(10), jnp.arange(10, 20))
    chunk_size = 4

    def pytree_func(a, b):
        return a + b

    expected = jax.vmap(lambda x: pytree_func(*x))(data)
    actual = chunked_map(pytree_func, data, chunk_size)

    chex.assert_trees_all_close(actual, expected)

def test_chunked_map_static_args():
    """Tests chunked_map with static arguments."""
    data = jnp.arange(8)
    chunk_size = 3
    static_arg = 5

    def func_with_static_arg(x, y):
        return x + y

    expected = jax.vmap(lambda x: func_with_static_arg(x, static_arg))(data)
    actual = chunked_map(func_with_static_arg, data, chunk_size, static_args={'y': static_arg})

    chex.assert_trees_all_close(actual, expected)

def test_chunked_map_zero_size_input():
    """Tests chunked_map with a zero-size leading axis."""
    data = jnp.zeros((0, 5))
    chunk_size = 3

    expected = jax.vmap(simple_func)(data)
    actual = chunked_map(simple_func, data, chunk_size)

    chex.assert_trees_all_close(actual, expected)