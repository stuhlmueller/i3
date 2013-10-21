import math
import pytest

from i3 import exact_inference
from i3 import invert
from i3 import random_world
from i3 import train
from i3 import utils
from i3.networks import sprinkler_net


class TestSprinklerBayesNet(object):
  """Test training on a three-node rain/sprinkler/grass network."""
  
  def setup(self):
    self.rng = utils.RandomState(seed=0)
    self.net = sprinkler_net.get(self.rng)

  @pytest.mark.parametrize("evidence_index", [0, 1, 2, 3])    
  def test(self, evidence_index):
    evidence = sprinkler_net.evidence(evidence_index)
    evidence_nodes = [self.net.nodes_by_index[index]
                      for index in evidence.keys()]    
    inverse_map = invert.compute_inverse_map(
      self.net, evidence_nodes, self.rng)
    assert len(inverse_map) == 2
    trainer = train.Trainer(inverse_map)
    num_samples = 30000
    for _ in xrange(num_samples):
      sample = self.net.sample()
      trainer.observe(sample)
    trainer.finalize()
  
    # Compare marginal log probability for evidence node with prior marginals.
    empty_world = random_world.RandomWorld()
    enumerator = exact_inference.Enumerator(self.net, empty_world)
    exact_marginals = enumerator.marginals()
    for evidence_node in evidence_nodes:
      for value in [0, 1]:
        log_prob_true = math.log(exact_marginals[evidence_node.index][value])
        for inverse_net in inverse_map.values():
          log_prob_empirical = inverse_net.nodes_by_index[
            evidence_node.index].log_probability(empty_world, value)
          print abs(log_prob_true - log_prob_empirical)
          assert abs(log_prob_true - log_prob_empirical) < .02
  
    # For each inverse network, take unconditional samples, compare
    # marginals to prior network.
    num_samples = 30000
    for inverse_net in inverse_map.values():
      counts = [[0, 0], [0, 0], [0, 0]]
      for _ in xrange(num_samples):
        world = inverse_net.sample()
        for (index, value) in world.items():
          counts[index][value] += 1
      for index in [0, 1, 2]:
        true_dist = enumerator.marginalize_node(
          self.net.nodes_by_index[index])
        empirical_dist = utils.normalize(counts[index])
        for (p_true, p_empirical) in zip(true_dist, empirical_dist):
          print abs(p_true - p_empirical)
          assert abs(p_true - p_empirical) < .02
        
      
