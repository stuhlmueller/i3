"""Tests for sampling and probability calculation utilities."""

import math
import numpy as np
import pytest

from i3 import utils


def test_safelog_base():
  """Test that safe_log does not differ from log for values > 0."""
  nums = [1, 10, 100, 1000, 10000]
  np.testing.assert_array_almost_equal(
    [utils.safe_log(num) for num in nums],
    [math.log(num) for num in nums])


def test_safelog_extended():
  """Test that safe_log returns -inf for 0."""
  assert utils.safe_log(0) == utils.NEGATIVE_INFINITY
  assert utils.NEGATIVE_INFINITY == float("-inf")


def test_significantly_greater():
  """Test that t-test works."""
  alpha = 0.01
  a = [9, 10, 11, 10]
  b = [1, 2, 1, 0]
  assert utils.significantly_greater(a, b, alpha)
  assert not utils.significantly_greater(a, a, alpha)
  assert not utils.significantly_greater(b, b, alpha)
  c = [9, 8, 9, 10]
  assert not utils.significantly_greater(a, c, alpha)
  assert not utils.significantly_greater(c, a, alpha)
  assert utils.significantly_greater(c, b, alpha)
  assert not utils.significantly_greater(b, c, alpha)


class TestRandomState(object):
  """Test RandomState class."""

  def setup(self):
    self.rng_1 = utils.RandomState(seed=0)
    self.rng_2 = utils.RandomState(seed=0)

  def test_categorical_a(self):
    """Test categorical sampler."""
    num_samples = 100
    sampler_1 = self.rng_1.categorical_sampler(["a", "b"], [0.5, 0.5])
    samples_1 = [sampler_1() for _ in xrange(num_samples)]
    sampler_2 = self.rng_2.categorical_sampler(["a", "b"], [0.5, 0.5])
    samples_2 = [sampler_2() for _ in xrange(num_samples)]
    assert samples_1 == samples_2
    for value in ["a", "b"]:
      for samples in [samples_1, samples_2]:
        assert samples.count(value) > 10
    sampler_3 = self.rng_2.categorical_sampler(["a"], [1.0])
    samples_3 = [sampler_3() for _ in xrange(num_samples)]
    assert samples_3 == ["a"] * num_samples

  def test_categorical_b(self):
    with pytest.raises(ValueError):
      self.rng_2.categorical_sampler([], [])
    with pytest.raises(ValueError):
      self.rng_2.categorical_sampler([], [1.0])
    with pytest.raises(ValueError):
      self.rng_2.categorical_sampler(["A", "B"], [1.0])
    with pytest.raises(ValueError):
      self.rng_2.categorical_sampler(["A"], [0.5, 0.5])

  def test_categorical_c(self):
    num_samples = 100000
    sampler_7 = self.rng_2.categorical_sampler(["A", "B", "C"], [0.1, 0.3, 0.6])
    samples_7 = [sampler_7() for _ in xrange(num_samples)]
    utils.assert_in_interval(samples_7.count("A"), 0.1, num_samples)
    utils.assert_in_interval(samples_7.count("B"), 0.3, num_samples)
    utils.assert_in_interval(samples_7.count("C"), 0.6, num_samples)

  def test_categorical_d(self):
    num_samples = 100000
    sampler_8 = self.rng_2.categorical_sampler(range(10), [0.1] * 10)
    samples_8 = [sampler_8() for _ in xrange(num_samples)]
    for i in range(10):
      utils.assert_in_interval(samples_8.count(i), 0.1, num_samples)

  def test_random_permutation(self):
    """Test shuffle functionality."""
    array_1 = self.rng_1.random_permutation(5)
    array_2 = self.rng_2.random_permutation(5)
    np.testing.assert_array_almost_equal(array_1, array_2)
    for i in range(5):
      assert i in array_1
    array_3 = [str(i) + "!" for i in range(10000)]
    array_4 = self.rng_1.random_permutation(array_3)
    array_5 = self.rng_2.random_permutation(array_3)
    np.testing.assert_array_equal(array_4, array_5)
    for value in array_3:
      assert value in array_4
    assert array_3[0] != array_4[0]

  def test_flip(self):
    """Test Boolean coin."""
    num_samples = 10000
    samples = [self.rng_1.flip(.3) for _ in xrange(num_samples)]
    utils.assert_in_interval(samples.count(True), 0.3, num_samples)
    utils.assert_in_interval(samples.count(False), 0.7, num_samples)


def test_lexicographic_combinations():
  """Test ordered combinations."""
  mappings = (
    ([], [[]]),
    ([[], [1]], []),
    ([[1], []], []),
    ([[1], [2]], [[1, 2]]),
    ([[1], [2, 3]], [[1, 2], [1, 3]]),
    ([[1, 3], [2]], [[1, 2], [3, 2]])
  )
  for (input, output) in mappings:
    assert list(utils.lexicographic_combinations(input)) == output


def test_is_range():
  """Check function that tests if a list is a range."""
  mappings = (
    ([], 0, True),
    ([], 1, True),    
    ([0], 0, True),
    ([1], 1, True),
    ([0], 1, False),
    ([1], 0, False),
    ([0, 1], 0, True),
    ([1, 2], 1, True),
    ([0, 1], 1, False),
    ([1, 0], 0, False),
    ([0, 1, 2, 3, 4, 5, 6], 0, True),
    ([1, 2, 3, 4, 5, 6, 7], 1, True)
  )  
  for (input_list, input_start, output) in mappings:
    assert utils.is_range(input_list, start=input_start) == output
