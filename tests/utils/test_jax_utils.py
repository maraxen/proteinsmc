import jax
import jax.numpy as jnp
import pytest
import chex
from proteinsmc.utils.jax_utils import (
    chunked_map,
    generate_jax_uuid,
    generate_jax_hash,
)

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


class TestChunkedMapEdgeCases:
    """Test edge cases for chunked_map."""

    def test_chunked_map_large_chunk_size(self):
        """Test chunked_map with chunk size larger than data."""
        data = jnp.arange(10)
        chunk_size = 100

        expected = jax.vmap(simple_func)(data)
        actual = chunked_map(simple_func, data, chunk_size)

        chex.assert_trees_all_close(actual, expected)

    def test_chunked_map_chunk_size_one(self):
        """Test chunked_map with chunk size of 1."""
        data = jnp.arange(10)
        chunk_size = 1

        expected = jax.vmap(simple_func)(data)
        actual = chunked_map(simple_func, data, chunk_size)

        chex.assert_trees_all_close(actual, expected)

    def test_chunked_map_invalid_static_args(self):
        """Test chunked_map with invalid static_args type."""
        data = jnp.arange(10)
        chunk_size = 3

        with pytest.raises(TypeError, match="static_args must be a dictionary"):
            chunked_map(simple_func, data, chunk_size, static_args="invalid")  # type: ignore[arg-type]

    def test_chunked_map_multiple_static_args(self):
        """Test chunked_map with multiple static arguments."""
        data = jnp.arange(8)
        chunk_size = 3

        def func_with_multiple_args(x, y, z):
            return x + y * z

        expected = jax.vmap(lambda x: func_with_multiple_args(x, 2, 3))(data)
        actual = chunked_map(
            func_with_multiple_args, data, chunk_size, static_args={'y': 2, 'z': 3}
        )

        chex.assert_trees_all_close(actual, expected)

    def test_chunked_map_nested_pytree(self):
        """Test chunked_map with nested PyTree structure."""
        data = {
            'a': jnp.arange(10),
            'b': {'c': jnp.arange(10, 20), 'd': jnp.arange(20, 30)}
        }
        chunk_size = 4

        def nested_func(tree):
            return tree['a'] + tree['b']['c'] + tree['b']['d']

        expected = jax.vmap(nested_func)(data)
        actual = chunked_map(nested_func, data, chunk_size)

        chex.assert_trees_all_close(actual, expected)

    def test_chunked_map_preserves_dtype(self):
        """Test that chunked_map preserves data types."""
        data = jnp.arange(10, dtype=jnp.float32)
        chunk_size = 3

        def float_func(x):
            return x * 2.5

        result = chunked_map(float_func, data, chunk_size)
        assert result.dtype == jnp.float32

    def test_chunked_map_multidimensional_input(self):
        """Test chunked_map with multidimensional input."""
        data = jnp.arange(60).reshape(10, 6)
        chunk_size = 3

        def sum_func(x):
            return jnp.sum(x)

        expected = jax.vmap(sum_func)(data)
        actual = chunked_map(sum_func, data, chunk_size)

        chex.assert_trees_all_close(actual, expected)


class TestGenerateJaxUUID:
    """Test UUID generation functions."""

    def test_generate_jax_uuid_shape(self):
        """Test that UUID has correct shape."""
        key = jax.random.PRNGKey(42)
        uuid_array, new_key = generate_jax_uuid(key)

        assert uuid_array.shape == (16,)
        assert uuid_array.dtype == jnp.uint8

    def test_generate_jax_uuid_range(self):
        """Test that UUID values are in valid range."""
        key = jax.random.PRNGKey(42)
        uuid_array, new_key = generate_jax_uuid(key)

        assert jnp.all(uuid_array >= 0)
        assert jnp.all(uuid_array <= 255)

    def test_generate_jax_uuid_updates_key(self):
        """Test that UUID generation updates PRNG key."""
        key = jax.random.PRNGKey(42)
        uuid_array, new_key = generate_jax_uuid(key)

        # Keys should be different
        assert not jnp.array_equal(key, new_key)

    def test_generate_jax_uuid_determinism(self):
        """Test that same key produces same UUID."""
        key = jax.random.PRNGKey(42)
        uuid1, _ = generate_jax_uuid(key)
        uuid2, _ = generate_jax_uuid(key)

        chex.assert_trees_all_close(uuid1, uuid2)

    def test_generate_jax_uuid_randomness(self):
        """Test that different keys produce different UUIDs."""
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(43)

        uuid1, _ = generate_jax_uuid(key1)
        uuid2, _ = generate_jax_uuid(key2)

        # Should not be identical
        assert not jnp.all(uuid1 == uuid2)

    def test_generate_jax_uuid_sequence(self):
        """Test generating multiple UUIDs in sequence."""
        key = jax.random.PRNGKey(0)
        uuids = []

        for _ in range(5):
            uuid_array, key = generate_jax_uuid(key)
            uuids.append(uuid_array)

        # All UUIDs should be different
        for i in range(len(uuids)):
            for j in range(i + 1, len(uuids)):
                assert not jnp.all(uuids[i] == uuids[j])


class TestGenerateJaxHash:
    """Test hash-based UUID generation."""

    def test_generate_jax_hash_shape(self):
        """Test that hash has correct shape."""
        key = jax.random.PRNGKey(42)
        hash_array, new_key = generate_jax_hash(key, 123)

        # The current implementation returns (16, 2) due to jax.random.fold_in
        # returning a key shape, then extracting bytes
        assert hash_array.shape == (16, 2)
        assert hash_array.dtype == jnp.uint8

    def test_generate_jax_hash_range(self):
        """Test that hash values are in valid range."""
        key = jax.random.PRNGKey(42)
        hash_array, new_key = generate_jax_hash(key, 123)

        assert jnp.all(hash_array >= 0)
        assert jnp.all(hash_array <= 255)

    def test_generate_jax_hash_updates_key(self):
        """Test that hash generation updates PRNG key."""
        key = jax.random.PRNGKey(42)
        hash_array, new_key = generate_jax_hash(key, 123)

        # Keys should be different
        assert not jnp.array_equal(key, new_key)

    def test_generate_jax_hash_determinism_same_data(self):
        """Test that same key and data produce same hash."""
        key = jax.random.PRNGKey(42)
        hash1, _ = generate_jax_hash(key, 123)
        hash2, _ = generate_jax_hash(key, 123)

        chex.assert_trees_all_close(hash1, hash2)

    def test_generate_jax_hash_different_data(self):
        """Test that different data produces different hashes."""
        key = jax.random.PRNGKey(42)
        hash1, _ = generate_jax_hash(key, 123)
        hash2, _ = generate_jax_hash(key, 456)

        # Should not be identical
        assert not jnp.all(hash1 == hash2)

    def test_generate_jax_hash_different_keys_same_data(self):
        """Test that different keys with same data produce different hashes."""
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(43)

        hash1, _ = generate_jax_hash(key1, 123)
        hash2, _ = generate_jax_hash(key2, 123)

        # Should not be identical
        assert not jnp.all(hash1 == hash2)

    def test_generate_jax_hash_zero_data(self):
        """Test hash generation with zero data."""
        key = jax.random.PRNGKey(42)
        hash_array, new_key = generate_jax_hash(key, 0)

        assert hash_array.shape == (16, 2)
        assert hash_array.dtype == jnp.uint8

    def test_generate_jax_hash_positive_data(self):
        """Test hash generation with positive data values."""
        key = jax.random.PRNGKey(42)
        hash_array, new_key = generate_jax_hash(key, 10**9)

        assert hash_array.shape == (16, 2)
        assert hash_array.dtype == jnp.uint8

