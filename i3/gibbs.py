"""Algorithsm for computing Gibbs distributions (conditionals)."""

import math

from i3 import distribution
from i3 import random_world


def gibbs_probabilities(node, world):
  """Given values for Markov blanket of node, compute Gibbs probabilities.

  Args:
    node: a BayesNetNode
    world: a random world that includes all nodes in the Markov
      blanket of the node of interest

  Returns:
    a list of probabilities, one for each value in the support of node.
  """
  temp_world = world.copy()
  gibbs_probs = []
  for value in node.support:
    node_logprob = node.log_probability(world, value)
    temp_world[node] = value
    children_logprob = sum(
      child.log_probability(temp_world, world[child])
      for child in node.children)
    gibbs_probs.append(math.exp(node_logprob + children_logprob))
  return gibbs_probs


def gibbs_distribution(node, world, rng):
  """Given values for Markov blanket of node, compute Gibbs distribution.

  Args:
    node: a BayesNetNode
    world: a random world that includes all nodes in the Markov
      blanket of the node of interest

  Returns:
    a categorical distribution on the support of node
  """
  gibbs_probs = gibbs_probabilities(node, world)
  gibbs_dist = distribution.CategoricalDistribution(
    node.support, gibbs_probs, rng)
  return gibbs_dist


def all_gibbs_distributions(node, rng):
  """Get mapping from Markov blanket vals (sorted by var) to Gibbs dists."""
  markov_blanket_nodes = sorted(node.markov_blanket)
  gibbs_dists = {}
  for world in random_world.all_random_worlds(
      markov_blanket_nodes, node.net.node_count):
    gibbs_dist = gibbs_distribution(node, world, rng)
    markov_blanket_vals = tuple(
      [world[n] for n in markov_blanket_nodes])
    gibbs_dists[markov_blanket_vals] = gibbs_dist
  return gibbs_dists        
