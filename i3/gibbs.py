"""Algorithsm for computing Gibbs distributions (conditionals)."""

import math

from i3 import distribution
from i3 import utils


def gibbs_probabilities(node, random_world):
  """Given values for Markov blanket of node, compute Gibbs probabilities.

  Args:
    node: a BayesNetNode
    random_world: a mapping from nodes to values that includes all
      nodes in the Markov blanket of the node of interest

  Returns:
    a list of probabilities, one for each value in the support of node.
  """
  support = node.support(random_world)
  if len(support) == 1:
    return [1.0]
  
  coparents = []
  for child in node.children:
    for parent in child.parents:
      if parent != node:
        coparents.append(parent)
  coparents = set(coparents)
  coparent_random_world = dict(
    (coparent, random_world[coparent]) for coparent in coparents)

  gibbs_probs = []
  
  for value in support:
    node_logprob = node.log_probability(random_world, value)
    coparent_random_world[node] = value
    children_logprob = sum(
      child.log_probability(coparent_random_world, random_world[child])
      for child in node.children)
    gibbs_probs.append(math.exp(node_logprob + children_logprob))
  
  return gibbs_probs


def gibbs_distribution(node, random_world, rng):
  """Given values for Markov blanket of node, compute Gibbs distribution.

  Args:
    node: a BayesNetNode
    random_world: a mapping from nodes to values that includes all
      nodes in the Markov blanket of the node of interest

  Returns:
    a categorical distribution on the support of node
  """
  gibbs_probs = gibbs_probabilities(node, random_world)
  gibbs_dist = distribution.CategoricalDistribution(
    node.support(random_world), gibbs_probs, rng)
  return gibbs_dist


def all_gibbs_distributions(node, support, rng):
  """Get mapping from Markov blanket vals (sorted by var) to Gibbs dists."""
  markov_blanket_vars = sorted(node.markov_blanket())
  gibbs_dists = {}
  for random_world in utils.all_random_worlds(markov_blanket_vars, support):
    gibbs_dist = gibbs_distribution(node, random_world, rng)
    markov_blanket_vals = tuple(
      [random_world[var] for var in markov_blanket_vars])
    gibbs_dists[markov_blanket_vals] = gibbs_dist
  return gibbs_dists        
