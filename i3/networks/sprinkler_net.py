"""A discrete three-node network."""

from i3 import bayesnet
from i3 import distribution


def get(rng):
  rain_node = bayesnet.BayesNetNode(
    name="Rain",
    parents=[],
    get_distribution=lambda: distribution.CategoricalDistribution(
      [True, False], [.2, .8], rng),
    full_support=[True, False])
  sprinkler_node = bayesnet.BayesNetNode(
    name="Sprinkler",
    parents=[rain_node],
    get_distribution=lambda rain: distribution.CategoricalDistribution(
      [True, False],
      [0.01, 0.99] if rain else [0.4, 0.6],
      rng),
    full_support=[True, False])
  rain_node.add_child(sprinkler_node)
  grass_cpt = {
    (False, False): [0, 1],
    (False, True): [0.8, 0.2],
    (True, False): [0.9, 0.1],
    (True, True): [0.99, 0.01]
  }
  grass_node = bayesnet.BayesNetNode(
    name="Grass",
    parents=[rain_node, sprinkler_node],
    get_distribution=lambda rain, sprinkler: (
      distribution.CategoricalDistribution(
        [True, False],
        grass_cpt[(sprinkler, rain)],
        rng)),
    full_support=[True, False])
  sprinkler_node.add_child(grass_node)
  rain_node.add_child(grass_node)
  net = bayesnet.BayesNet([rain_node, sprinkler_node, grass_node])
  return net
