"""Test MCMC on imported UAI networks."""
from __future__ import division

import cProfile
import numpy as np

from i3 import marginals
from i3 import mcmc
from i3 import utils
from i3.networks import triangle_net


class TestTriangleNetwork(object):

  def setup(self):
    self.rng = utils.RandomState(seed=0)
    self.net = triangle_net.get(self.rng)
    self.evidence = triangle_net.evidence()
    self.marginals = triangle_net.marginals()

  def test_gibbs(self):
    chain = mcmc.GibbsChain(self.net, self.rng, self.evidence)
    chain.initialize_state()
    empirical_marginals = chain.marginals(10000)
    diff = self.marginals - empirical_marginals
    average_error = sum(diff.values())/len(diff)
    print average_error
    assert average_error < .05


def run_test():
  t = TestTriangleNetwork()
  t.setup()
  t.test_gibbs()


if __name__ == "__main__":
  cProfile.run("run_test()", sort="cumulative")
