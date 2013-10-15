"""Tests for exact Bayes net inference."""

import numpy as np

from i3 import exact_inference
from i3 import random_world
from i3 import utils
from i3.networks import sprinkler_net


class TestSprinklerBayesNet(object):
  """Test enumeration on a three-node rain/sprinkler/grass network."""

  def setup(self):
    """Set up random stream and sprinkler network."""
    rng = utils.RandomState(seed=0)
    self.net = sprinkler_net.get(rng)

  def test_enumerate(self):
    """Check that computed marginal probability is correct."""
    grass_node = self.net.find_node("Grass")
    rain_node = self.net.find_node("Rain")
    evidence = random_world.RandomWorld([grass_node], [1])
    enumerator = exact_inference.Enumerator(self.net, evidence)
    inferred_dist = enumerator.marginalize_node(rain_node)
    np.testing.assert_almost_equal(inferred_dist[1], 0.24277141298417898)
