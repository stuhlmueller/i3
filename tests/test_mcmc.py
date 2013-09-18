"""Tests for Bayes net samplers."""

from i3 import mcmc
from i3 import utils
from i3.networks import sprinkler_net


class TestRejection(object):
  """Test rejection sampler."""

  def setup(self):
    """Create reandom stream and sprinkler network."""
    self.rng = utils.RandomState(seed=1)
    self.net = sprinkler_net.get(self.rng)

  def test_sprinkler(self):
    """Check that inference result is within +/- 100 of truth."""
    grass_node = self.net.find_node("Grass")
    rain_node = self.net.find_node("Rain")
    evidence = {
      grass_node : True
    }
    rejection_chain = mcmc.RejectionChain(self.net, self.rng, evidence)
    rejection_chain.initialize_state()
    rain_count = 0
    for _ in xrange(10000):
      rejection_chain.transition()
      if rejection_chain.state[rain_node]:
        rain_count += 1
    assert 3477 < rain_count < 3677
    
