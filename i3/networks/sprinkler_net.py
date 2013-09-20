"""A discrete three-node network."""

from i3 import bayesnet


def get(rng):
  rain_node = bayesnet.CategoricalNode(
    name="Rain",
    values=[True, False],
    get_probabilities=lambda: [.2, .8],
    rng=rng)
  
  sprinkler_node = bayesnet.CategoricalNode(
    name="Sprinkler",
    values=[True, False],
    get_probabilities=lambda rain: [0.01, 0.99] if rain else [0.4, 0.6],
    rng=rng)

  def get_grass_probabilities(rain, sprinkler):
    grass_cpt = {
      (False, False): [0.8, 0.2],
      (False, True): [0.7, 0.3],
      (True, False): [0.85, 0.15],
      (True, True): [0.9, 0.1]
    }
    return grass_cpt[(sprinkler, rain)]
    
  grass_node = bayesnet.CategoricalNode(
    name="Grass",
    values=[True, False],
    get_probabilities=get_grass_probabilities,
    rng=rng)

  nodes = [rain_node, sprinkler_node, grass_node]
  edges = [(rain_node, sprinkler_node),
           (rain_node, grass_node),
           (sprinkler_node, grass_node)]
  net = bayesnet.BayesNet("Sprinkler network", nodes, edges)
  return net
