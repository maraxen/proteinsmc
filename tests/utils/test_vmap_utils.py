import jax.numpy as jnp

from proteinsmc.utils.vmap_utils import chunked_vmap


def test_chunked_vmap_basic_add():
  def add_one(x):
    return x + 1

  data = jnp.arange(10)
  out = chunked_vmap(add_one, data, chunk_size=3)
  expected = data + 1
  assert jnp.allclose(out, expected)


def test_chunked_vmap_2d_array():
  def double(x):
    return x * 2

  data = jnp.arange(20).reshape(10, 2)
  out = chunked_vmap(double, data, chunk_size=4)
  expected = data * 2
  assert jnp.allclose(out, expected)


def test_chunked_vmap_with_static_args():
  def add_static(x, static):
    return x + static

  data = jnp.arange(8)
  static = 5
  out = chunked_vmap(add_static, data, chunk_size=2, static_args=static)
  expected = data + static
  assert jnp.allclose(out, expected)


def test_chunked_vmap_pytree():
  def add_tuple(x):
    a, b = x
    return a + 1, b * 2

  data = (jnp.arange(6), jnp.arange(6, 12))
  out = chunked_vmap(add_tuple, data, chunk_size=2)
  expected = (data[0] + 1, data[1] * 2)
  assert jnp.allclose(out[0], expected[0])
  assert jnp.allclose(out[1], expected[1])


def test_chunked_vmap_empty_data():
  data = jnp.array([])

  def f(x):
    return x + 1

  out = chunked_vmap(f, data, chunk_size=3)
  assert out.shape == (0,)


def test_chunked_vmap_chunk_size_larger_than_data():
  data = jnp.arange(5)

  def f(x):
    return x * 3

  out = chunked_vmap(f, data, chunk_size=10)
  expected = data * 3
  assert jnp.allclose(out, expected)


def test_chunked_vmap_non_divisible_chunk():
  data = jnp.arange(7)

  def f(x):
    return x - 2

  out = chunked_vmap(f, data, chunk_size=3)
  expected = data - 2
  assert jnp.allclose(out, expected)
