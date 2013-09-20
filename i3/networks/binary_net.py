"""A two-node Bayes net with deterministic CPTs."""

from i3 import bayesnet


def get(rng):
  """Return binary Bayes net."""
  node_1 = bayesnet.CategoricalNode(
    name="node_1",
    values=[True],
    get_probabilities=lambda: [1.0],
    rng=rng)
  node_2 = bayesnet.CategoricalNode(
    name="node_2",
    values=[True, False],
    get_probabilities=lambda n1: [0.0, 1.0] if n1 else [1.0, 0.0],
    rng=rng)
  net = bayesnet.BayesNet(
    "Binary network",
    nodes=[node_1, node_2],
    edges=[(node_1, node_2)])
  return net
  
