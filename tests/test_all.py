"""Test the complete sample/train/learn/test pipeline."""

from __future__ import division

import pytest

from i3 import exact_inference
from i3 import mcmc
from i3 import invert
from i3 import train
from i3 import utils
from i3.networks import sprinkler_net


def run_test(rng, net, evidence, proposal_size):
  
  num_samples = 100000
  
  evidence_nodes = [net.nodes_by_index[evidence.keys()[0]]]

  print "Computing trainable inverse nets..."
  inverse_map = invert.compute_inverse_map(
    net, evidence_nodes, rng, max_inverse_size=proposal_size)

  print "Initializing trainer..."
  trainer = train.Trainer(net, inverse_map, precompute_gibbs=False)

  print "Training..."
  training_sampler = mcmc.GibbsChain(net, rng, evidence)
  training_sampler.initialize_state()
  for _ in xrange(num_samples):
    training_sampler.transition()
    trainer.observe(training_sampler.state)

  print "Finishing training..."
  trainer.finalize()  # Does not include deterministic Gibbs yet!

  print "Computing exact solution..."
  enumerator = exact_inference.Enumerator(net, evidence)
  true_marginals = enumerator.marginals()

  # The following works even when we don't compute full inverse
  # networks, since even incomplete networks still contain all nodes,
  # which allows us to learn marginals.
  print "Testing (exact inference)..."
  for inverse_net in inverse_map.values():
    enumerator = exact_inference.Enumerator(inverse_net, evidence)
    inverse_marginals = enumerator.marginals()
    print true_marginals - inverse_marginals
    assert true_marginals - inverse_marginals < .01
  
  print "Testing (sampling)..."
  test_sampler = mcmc.InverseChain(
    net, inverse_map, rng, evidence, proposal_size=proposal_size)
  test_sampler.initialize_state()
  inverse_marginals = test_sampler.marginals(num_samples)

  print true_marginals - inverse_marginals
  assert true_marginals - inverse_marginals < .02


@pytest.mark.slow
@pytest.mark.parametrize(
  ("proposal_size", "evidence_index"),
  utils.lexicographic_combinations([[1, 2, 3], [0, 1, 2, 3]]))
def test_sprinkler_net(proposal_size, evidence_index):
  rng = utils.RandomState(seed=0)
  net = sprinkler_net.get(rng)
  evidence = sprinkler_net.evidence(evidence_index)
  run_test(rng, net, evidence, proposal_size=proposal_size)


if __name__ == "__main__":
  test_sprinkler_net(proposal_size=2, evidence_index=2)
