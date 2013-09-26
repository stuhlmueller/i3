"""Test probability distributions."""

import math
import numpy as np

from i3 import distribution
from i3 import utils


def test_categorical_distribution():
  """Test categorical distribution."""
  rng = utils.RandomState(0)
  dist = distribution.CategoricalDistribution(
    values=["a", "b"],
    probabilities=[.3, .7],
    rng=rng)
  samples = [dist.sample() for _ in range(10000)]
  utils.assert_in_interval(samples.count("a"), .3, 10000, .95)
  utils.assert_in_interval(samples.count("b"), .7, 10000, .95)
  np.testing.assert_almost_equal(
    .3, math.exp(dist.log_probability("a")))
  np.testing.assert_almost_equal(
    .7, math.exp(dist.log_probability("b")))  
  assert sorted(dist.support()) == ["a", "b"]
