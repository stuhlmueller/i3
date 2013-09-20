"""Tests for exact Bayes net inference."""

import numpy as np

from i3 import exact_inference
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
    evidence = {grass_node: True}
    enumerator = exact_inference.Enumerator(self.net)
    inferred_dist = enumerator.marginalize(evidence, rain_node)
    np.testing.assert_almost_equal(inferred_dist[True], 0.17629331994)
