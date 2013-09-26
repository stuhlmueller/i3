"""Test import of UAI Bayes net files."""

import numpy as np

from i3 import random_world
from i3 import uai_import
from i3 import utils


NETWORK_STRING = """BAYES
3
2 2 3
3
1 0
2 0 1
2 1 2

2
 0.436 0.564

4
 0.128 0.872
 0.920 0.080

6
 0.210 0.333 0.457
 0.811 0.000 0.189"""

NETWORK_PROBABILITIES = [
  ([0, 0, 0], 0.436 * 0.128 * 0.210),
  ([1, 1, 2], 0.564 * 0.080 * 0.189),
  ([1, 0, 2], 0.564 * 0.920 * 0.457),
  ([0, 0, 1], 0.436 * 0.128 * 0.333),
  ([0, 1, 1], 0.436 * 0.872 * 0.000),        
]


def test_uai_import():
  """Check that probabilities for imported Bayes net are as expected."""
  net = uai_import.network_from_string(NETWORK_STRING, utils.RandomState())
  assert len(net.sample()) == 3
  for (values, prob) in NETWORK_PROBABILITIES:
    print values, prob
    world = random_world.RandomWorld(
      nodes=net.nodes_by_index,
      values=values)
    np.testing.assert_almost_equal(
      net.log_probability(world),
      utils.safe_log(prob))
