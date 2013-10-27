import pytest

from i3 import exact_inference
from i3 import invert
from i3 import mcmc
from i3 import random_world
from i3 import train
from i3 import utils
from i3.networks import sprinkler_net


class TestInverseChain(object):

  def setup(self):
    self.rng = utils.RandomState(seed=0)
    self.net = sprinkler_net.get(self.rng)

  @pytest.mark.slow
  @pytest.mark.parametrize(
    ("proposal_size", "evidence_index"),
    utils.lexicographic_combinations([[1, 2, 3], [0, 1, 2, 3]]))
  def test_inverse_sampler(self, proposal_size, evidence_index):
    evidence = sprinkler_net.evidence(evidence_index)
    evidence_nodes = [self.net.nodes_by_index[index]
                      for index in evidence.keys()]
    inverse_map = invert.compute_inverse_map(
      self.net, evidence_nodes, self.rng)
    # 2. Initialize inverses with uniform distributions
    trainer = train.Trainer(self.net, inverse_map, precompute_gibbs=False)
    # 3. Generate random data
    for _ in xrange(3):
      world = random_world.RandomWorld(
        range(self.net.node_count),
        [self.rng.choice(node.support)
         for node in self.net.nodes_by_index]
      )
      trainer.observe(world)
    trainer.finalize()
    # 4. Compute true answer
    enumerator = exact_inference.Enumerator(self.net, evidence)
    true_marginals = enumerator.marginals()
    # 5. Compute answer using inverse sampling
    num_samples = 50000
    test_sampler = mcmc.InverseChain(
      self.net, inverse_map, self.rng, evidence, proposal_size)
    test_sampler.initialize_state()
    inverse_marginals = test_sampler.marginals(num_samples)
    assert true_marginals - inverse_marginals < .02
