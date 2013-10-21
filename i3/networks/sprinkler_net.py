"""A discrete three-node network."""

from i3 import bayesnet
from i3 import evid


def get(rng):
  """Return three-node sprinkler Bayes net."""
  rain_node = bayesnet.TableBayesNetNode(
    index=0,
    domain_size=2,
    cpt_probabilities=[.8, .2],
    name="Rain")
  sprinkler_node = bayesnet.TableBayesNetNode(
    index=1,
    domain_size=2,
    cpt_probabilities=[
      0.01, 0.99,
      0.6, 0.4],
    name="Sprinkler")
  grass_node = bayesnet.TableBayesNetNode(
    index=2,
    domain_size=2,
    cpt_probabilities=[
      0.9, 0.1,
      0.3, 0.7,
      0.15, 0.85,
      0.05, 0.95],
    name="Grass")
  nodes = [rain_node, sprinkler_node, grass_node]
  edges = [(rain_node, sprinkler_node),
           (rain_node, grass_node),
           (sprinkler_node, grass_node)]
  net = bayesnet.BayesNet(
    rng=rng,
    nodes=nodes,
    edges=edges)
  net.compile()
  return net


def evidence(index):
  if index == 0:
    key, value = 2, 1
  elif index == 1:
    key, value = 2, 0
  elif index == 2:
    key, value = 1, 0
  elif index == 3:
    key, value = 1, 1
  else:
    raise ValueError("Unknown evidence index.")
  return evid.Evidence(keys=[key], values=[value])
