"""Test MCMC on imported UAI networks."""
from __future__ import division

import cProfile
import datetime
import pytest

from i3 import invert
from i3 import marg
from i3 import mcmc
from i3 import train
from i3 import utils
from i3.networks import triangle_net


class TestTriangleNetwork(object):

  def setup(self, determinism=95, seed=None):
    seed = 0 if seed is None else seed
    self.rng = utils.RandomState(seed)
    self.net = triangle_net.get(self.rng, determinism)
    self.evidence = triangle_net.evidence(0, determinism)
    self.evidence_nodes = [
      self.net.nodes_by_index[index] for index in self.evidence.keys()]
    self.num_latent_nodes = len(self.net.nodes()) - len(self.evidence_nodes)
    self.marginals = triangle_net.marginals(0, determinism)

  def test_gibbs(self):
    num_test_samples = 5000
    gibbs_chain = mcmc.GibbsChain(self.net, self.rng, self.evidence)
    gibbs_chain.initialize_state()
    gibbs_marginals = gibbs_chain.marginals(num_test_samples)
    gibbs_error = (self.marginals - gibbs_marginals).mean()
    print "Error (Gibbs): {}".format(gibbs_error)
    assert gibbs_error < .05

  def train_inverses(self, inverse_map, num_training_samples, precompute_gibbs):
    trainer = train.Trainer(self.net, inverse_map, precompute_gibbs)
    training_sampler = mcmc.GibbsChain(self.net, self.rng, self.evidence)
    training_sampler.initialize_state()
    counter = marg.MarginalCounter(self.net)
    for _ in xrange(num_training_samples):
      training_sampler.transition()
      trainer.observe(training_sampler.state)
      counter.observe(training_sampler.state)
    trainer.finalize()
    training_error = (self.marginals - counter.marginals()).mean()
    return training_error

  def check_inverses_by_samples(self, inverse_map, max_inverse_size,
                                num_test_samples):
    test_sampler = mcmc.InverseChain(
      self.net, inverse_map, self.rng, self.evidence, max_inverse_size)
    test_sampler.initialize_state()
    counter = marg.MarginalCounter(self.net)
    num_proposals_accepted = 0
    for _ in xrange(num_test_samples):
      accept = test_sampler.transition()
      counter.observe(test_sampler.state)
      num_proposals_accepted += accept
    inverse_marginals = counter.marginals()
    inverses_error = (self.marginals - inverse_marginals).mean()
    return inverses_error, num_proposals_accepted

  def check_inverses_by_time(self, inverse_map, max_inverse_size, test_seconds):
    test_sampler = mcmc.InverseChain(
      self.net, inverse_map, self.rng, self.evidence, max_inverse_size)
    test_sampler.initialize_state()
    counter = marg.MarginalCounter(self.net)
    num_proposals_accepted = 0
    start_time = datetime.datetime.now()
    num_proposals = 0
    while (datetime.datetime.now() - start_time).seconds < test_seconds:
      accept = test_sampler.transition()
      counter.observe(test_sampler.state)
      num_proposals_accepted += accept
      num_proposals += self.num_latent_nodes
    inverse_marginals = counter.marginals()
    inverses_error = (self.marginals - inverse_marginals).mean()
    return inverses_error, num_proposals, num_proposals_accepted

  @pytest.mark.slow
  @pytest.mark.parametrize(
    "precompute_gibbs,max_inverse_size",
    utils.lexicographic_combinations([[True, False], [1, 2]]))
  def test_inverses_error(self, precompute_gibbs, max_inverse_size):
    """Verify that error in estimated inverse marginals is low."""
    num_training_samples = 50000
    num_test_samples = 10000

    print "Computing inverse nets..."
    inverse_map = invert.compute_inverse_map(
      self.net, self.evidence_nodes, self.rng, max_inverse_size)

    print "Training on Gibbs samples..."
    training_error = self.train_inverses(
      inverse_map, num_training_samples, precompute_gibbs)
    print "Error (training): {}".format(training_error)
    assert training_error < .01

    print "Testing (inverses)..."
    test_error, num_proposals_accepted = self.check_inverses_by_samples(
      inverse_map, max_inverse_size, num_test_samples)
    print "Error (inverses): {}".format(test_error)
    assert test_error < .03

    num_proposals = num_test_samples * self.num_latent_nodes
    print "Accepted {} out of {} proposals".format(
      num_proposals_accepted, num_proposals)
    
    if max_inverse_size == 1 and precompute_gibbs:
      # Check that all proposals are accepted.
      assert num_proposals_accepted == num_proposals

  @pytest.mark.slow
  def test_inverses_performance(self):
    """Verify that bigger proposal sizes can result in better performance.

    This test uses clock time, not number of samples, to decide how
    many test samples to take.
    """
    num_training_samples = 50000
    test_seconds = 10
    precompute_gibbs = True
    max_inverse_size = 8

    print "Computing inverse nets..."
    inverse_map = invert.compute_inverse_map(
      self.net, self.evidence_nodes, self.rng, max_inverse_size)

    print "Training on Gibbs samples..."
    training_error = self.train_inverses(
      inverse_map, num_training_samples, precompute_gibbs)
    print "Error (training): {}".format(training_error)
    assert training_error < .01

    print "Testing (inverses)..."
    for inverse_size in range(1, max_inverse_size + 1):
      test_error, num_proposals, num_accepted = self.check_inverses_by_time(
        inverse_map, inverse_size, test_seconds)
      print "Error (inverses, max inverse size {}): {}".format(
        inverse_size, test_error)
      print "Accepted {} out of {} proposals\n".format(
        num_accepted, num_proposals)

  def profile_inverses(self):
    num_training_samples = 5000
    test_seconds = 10
    precompute_gibbs = False
    max_inverse_size = 3
    start_time = datetime.datetime.now()
    inverse_map = invert.compute_inverse_map(
      self.net, self.evidence_nodes, self.rng, max_inverse_size)
    t1 = datetime.datetime.now()
    print "Time to compute inverse map: {}".format(t1 - start_time)
    self.train_inverses(inverse_map, num_training_samples, precompute_gibbs)
    t2 = datetime.datetime.now()
    print "Time to train inverses: {}".format(t2 - t1)
    self.check_inverses_by_time(inverse_map, max_inverse_size, test_seconds)
    t3 = datetime.datetime.now()
    print "Time for test sampling: {}".format(t3 - t2)


def main():
  t = TestTriangleNetwork()
  t.setup(determinism=95, seed=0)
  # t.test_inverses_performance()
  t.profile_inverses()  

if __name__ == "__main__":
  cProfile.run("main()", sort="cumulative")
  # main()
