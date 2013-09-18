"""Tests for sampling and probability calculation utilities."""

import math
import numpy as np

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

def test_random_state():
  """Test RandomState class."""
  num_samples = 100
  rng_1 = utils.RandomState(seed=0)
  sampler_1 = rng_1.categorical_sampler(["a", "b"], [0.5, 0.5])
  samples_1 = [sampler_1() for _ in xrange(num_samples)]
  rng_2 = utils.RandomState(seed=0)
  sampler_2 = rng_2.categorical_sampler(["a", "b"], [0.5, 0.5])
  samples_2 = [sampler_2() for _ in xrange(num_samples)]
  assert samples_1 == samples_2
  for value in ["a", "b"]:
    for samples in [samples_1, samples_2]:
      assert samples.count(value) > 10
  
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
