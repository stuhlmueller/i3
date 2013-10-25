"""Test MCMC on imported UAI networks."""
from __future__ import division

import cProfile
import pytest

from i3 import invert
from i3 import marg
from i3 import mcmc
from i3 import train
from i3 import utils
from i3.networks import triangle_net


class TestTriangleNetwork(object):
  
  def setup(self, smooth=95):
    self.rng = utils.RandomState(seed=1)
    self.net = triangle_net.get(self.rng, smooth=smooth)
    self.evidence = triangle_net.evidence(0, smooth=smooth)
    self.marginals = triangle_net.marginals(0, smooth=smooth)

  def test_gibbs(self):
    chain = mcmc.GibbsChain(self.net, self.rng, self.evidence)
    chain.initialize_state()
    empirical_marginals = chain.marginals(10000)
    diff = self.marginals - empirical_marginals
    average_error = sum(diff.values()) / len(diff)
    print average_error
    assert average_error < .05

  @pytest.mark.parametrize("precompute_gibbs", [True, False])
  def test_inverse_mcmc(self, max_inverse_size=1, num_training_samples=50000,
                        num_test_samples=10000, precompute_gibbs=False):
    evidence_nodes = [self.net.nodes_by_index[self.evidence.keys()[0]]]

    print "Computing inverse nets..."    
    inverse_map = invert.compute_inverse_map(
      self.net, evidence_nodes, self.rng, max_inverse_size=max_inverse_size)
    
    print "Training..."
    trainer = train.Trainer(
      self.net, inverse_map, precompute_gibbs=precompute_gibbs)
    training_sampler = mcmc.GibbsChain(self.net, self.rng, self.evidence)
    training_sampler.initialize_state()
    counter = marg.MarginalCounter(self.net)
    for _ in xrange(num_training_samples):
      training_sampler.transition()
      trainer.observe(training_sampler.state)
      counter.observe(training_sampler.state)
    trainer.finalize()
    diff = self.marginals - counter.marginals()
    average_error = sum(diff.values()) / len(diff)
    print "Training samples:", average_error

    print "Testing (inverses)..."    
    test_sampler = mcmc.InverseChain(
      self.net, inverse_map, self.rng, self.evidence,
      proposal_size=max_inverse_size)
    test_sampler.initialize_state()
    counter = marg.MarginalCounter(self.net)
    num_proposals_accepted = 0
    for _ in xrange(num_test_samples):
      accept = test_sampler.transition()
      counter.observe(test_sampler.state)
      num_proposals_accepted += accept
    inverse_marginals = counter.marginals()
    diff = self.marginals - inverse_marginals
    inverses_error = sum(diff.values()) / len(diff)
    print "Inverses:", inverses_error
    if max_inverse_size == 1 and precompute_gibbs:
      num_proposals = (
        num_test_samples * (len(self.net.nodes_by_index) - len(evidence_nodes)))
      assert num_proposals_accepted == num_proposals, num_proposals_accepted

    print "Testing (gibbs)"
    gibbs_sampler = mcmc.GibbsChain(self.net, self.rng, self.evidence)
    gibbs_sampler.initialize_state()
    gibbs_marginals = gibbs_sampler.marginals(num_test_samples)
    diff = self.marginals - gibbs_marginals
    gibbs_error = sum(diff.values()) / len(diff)
    print "Gibbs:", gibbs_error

    assert gibbs_error < .03
    assert inverses_error < .03    


def run_test():
  t = TestTriangleNetwork()
  t.setup(smooth=95)
  t.test_inverse_mcmc(
    max_inverse_size=1,
    num_training_samples=50000,
    num_test_samples=10000,
    precompute_gibbs=False)


if __name__ == "__main__":
  # cProfile.run("run_test()", sort="cumulative")
  run_test()
