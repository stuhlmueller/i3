"""A two-node Bayes net with deterministic CPTs."""

from i3 import bayesnet
from i3 import distribution


def get_network(rng):
  """Return binary Bayes net."""
  get_dist_1 = lambda: distribution.CategoricalDistribution(
    [True], [1.0], rng)
  node_1 = bayesnet.BayesNetNode("node_1", [], get_dist_1)
  get_dist_2 = lambda parent_value: distribution.CategoricalDistribution(
    [not(parent_value)], [1.0], rng)
  node_2 = bayesnet.BayesNetNode("node_2", [node_1], get_dist_2)
  node_1.add_child(node_2)
  net = bayesnet.BayesNet([node_1, node_2])
  return net
  
