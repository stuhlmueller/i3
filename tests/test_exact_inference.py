"""Tests for exact Bayes net inference."""

import numpy as np
import pytest

from i3 import exact_inference
from i3 import mcmc
from i3 import random_world
from i3 import utils
from i3.networks import sprinkler_net


class TestSprinklerBayesNet(object):
  """Test enumeration on a three-node rain/sprinkler/grass network."""

  def setup(self):
    """Set up random stream and sprinkler network."""
    self.rng = utils.RandomState(seed=0)
    self.net = sprinkler_net.get(self.rng)

  def test_enumerate(self):
    """Check that computed marginal probability is correct."""
    grass_node = self.net.find_node("Grass")
    rain_node = self.net.find_node("Rain")
    evidence = random_world.RandomWorld([grass_node], [1])
    enumerator = exact_inference.Enumerator(self.net, evidence)
    inferred_dist = enumerator.marginalize_node(rain_node)
    np.testing.assert_almost_equal(inferred_dist[1], 0.24277141298417898)

  @pytest.mark.parametrize("evidence_index", [0, 1, 2, 3])
  def test_rejection(self, evidence_index):
    """Compare enumeration results with rejection results."""
    evidence = sprinkler_net.evidence(evidence_index)
    enumerator = exact_inference.Enumerator(self.net, evidence)
    enum_marginals = enumerator.marginals()
    rejection_chain = mcmc.RejectionChain(self.net, self.rng, evidence)
    rej_marginals = rejection_chain.marginals(20000)
    for p_diff in (enum_marginals - rej_marginals).values():
      assert p_diff < .01
