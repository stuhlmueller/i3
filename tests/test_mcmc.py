"""Tests for Bayes net samplers."""

from i3 import mcmc
from i3 import utils
from i3.networks import sprinkler_net


class TestSprinkler(object):
  """Test samplers on sprinkler network."""

  def setup(self):
    """Create random stream and sprinkler network."""
    self.rng = utils.RandomState(seed=0)
    self.net = sprinkler_net.get(self.rng)

  def run_sprinkler(self, chain_class):
    """Check that inference result is within +/- 100 of truth."""
    grass_node = self.net.find_node("Grass")
    rain_node = self.net.find_node("Rain")
    evidence = {grass_node: True}
    chain = chain_class(self.net, self.rng, evidence)
    chain.initialize_state()
    rain_count = 0
    for _ in xrange(100000):
      chain.transition()
      if chain.state[rain_node]:
        rain_count += 1
    print chain_class, rain_count
    assert 33000 < rain_count < 38000

  def test_rejection(self):
    self.run_sprinkler(mcmc.RejectionChain)

  def test_gibbs(self):
    self.run_sprinkler(mcmc.GibbsChain)    
