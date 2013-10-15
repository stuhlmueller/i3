import math

from i3 import exact_inference
from i3 import invert
from i3 import random_world
from i3 import train
from i3 import utils
from i3.networks import sprinkler_net


def test_trainer():
  rng = utils.RandomState(seed=0)
  net = sprinkler_net.get(rng)
  evidence_nodes = [net.find_node("Grass")]
  inverse_map = invert.compute_inverse_map(net, evidence_nodes, rng)
  assert len(inverse_map) == 2
  trainer = train.Trainer(inverse_map)
  num_samples = 10000
  for _ in xrange(num_samples):
    sample = net.sample()
    trainer.observe(sample)
  trainer.finalize()

  # Compare marginal log probability for evidence node with prior marginals.
  empty_world = random_world.RandomWorld()
  enumerator = exact_inference.Enumerator(net, empty_world)
  marginals = enumerator.marginalize_node(net.find_node("Grass"))
  for value in [0, 1]:
    log_prob_true = math.log(marginals[value])
    for inverse_net in inverse_map.values():
      log_prob_empirical = inverse_net.find_node("Grass").log_probability(
        empty_world, value)
      assert abs(log_prob_true - log_prob_empirical) < .1

      # For each inverse network, take unconditional samples, compare
    # marginals to prior network.
  num_samples = 10000
  for inverse_net in inverse_map.values():
    counts = [[0, 0], [0, 0], [0, 0]]
    for _ in xrange(num_samples):
      world = inverse_net.sample()
      for (index, value) in world.items():
        counts[index][value] += 1
    for index in [0, 1, 2]:
      true_dist = enumerator.marginalize_node(net.nodes_by_index[index])
      empirical_dist = utils.normalize(counts[index])
      for (p_true, p_empirical) in zip(true_dist, empirical_dist):
        assert abs(p_true - p_empirical) < .1
      
      
