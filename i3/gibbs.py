"""Algorithsm for computing Gibbs distributions (conditionals)."""

import math

from i3 import dist
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
  coparents = []
  for child in node.children:
    for parent in child.parents:
      if parent != node:
        coparents.append(parent)
  coparents = set(coparents)
  coparent_world = random_world.RandomWorld(
    coparents, [world[coparent] for coparent in coparents])

  gibbs_probs = []

  for value in node.support:
    node_logprob = node.log_probability(world, value)
    coparent_world[node] = value
    children_logprob = sum(
      child.log_probability(coparent_world, world[child])
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
  gibbs_dist = dist.CategoricalDistribution(
    node.support, gibbs_probs, rng)
  return gibbs_dist


def all_gibbs_distributions(node, rng):
  """Get mapping from Markov blanket vals (sorted by var) to Gibbs dists."""
  markov_blanket_vars = sorted(node.markov_blanket)
  gibbs_dists = {}
  for world in random_world.all_random_worlds(markov_blanket_vars):
    gibbs_dist = gibbs_distribution(node, world, rng)
    markov_blanket_vals = tuple(
      [world[var] for var in markov_blanket_vars])
    gibbs_dists[markov_blanket_vals] = gibbs_dist
  return gibbs_dists        
