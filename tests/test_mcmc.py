"""Tests for Bayes net samplers."""

from i3 import exact_inference
from i3 import mcmc
from i3 import random_world
from i3 import utils
from i3.networks import sprinkler_net


class TestSprinkler(object):
  """Test samplers on sprinkler network."""

  def setup(self):
    """Create random stream and sprinkler network."""
    self.rng = utils.RandomState(seed=0)
    self.net = sprinkler_net.get(self.rng)

  def run_sprinkler(self, chain_class):
    """Check that inference result is close to truth."""
    grass_node = self.net.find_node("Grass")
    rain_node = self.net.find_node("Rain")
    evidence = random_world.RandomWorld([grass_node], [True])
    chain = chain_class(self.net, self.rng, evidence)
    chain.initialize_state()
    rain_count = 0
    N = 100000
    for _ in xrange(N):
      chain.transition()
      if chain.state[rain_node]:
        rain_count += 1
    enumerator = exact_inference.Enumerator(self.net)
    exact_dist = enumerator.marginalize(evidence, rain_node)
    rain_prob = exact_dist[True]
    print rain_prob, rain_count
    assert rain_prob*N - 1000 < rain_count <  rain_prob*N + 1000

  def test_rejection(self):
    self.run_sprinkler(mcmc.RejectionChain)

  def test_gibbs(self):
    self.run_sprinkler(mcmc.GibbsChain)
