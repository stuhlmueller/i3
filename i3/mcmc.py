"""Markov chain inference for Bayesian networks."""
from __future__ import division

from i3 import gibbs
from i3 import marg
from i3 import random_world
from i3 import utils


class MarkovChain(object):
  def __init__(self, net, rng):
    self.net = net
    self.rng = rng

  def initialize_state(self):
    raise NotImplementedError()

  def marginals(self, num_transitions):
    """Compute marginal distribution of nodes.
    
    Compute marginal distribution of Bayes net nodes by repeatedly
    applying self transition kernel and storing state counts.

    Args:
      num_transitions: number of transitions to apply

    Returns:
      empirical marginals
    """
    counts = marg.Marginals(
      self.net.nodes_by_index,
      [[0] * len(node.support) for node in self.net.nodes_by_index]
    )
    for i in xrange(num_transitions):
      self.transition()
      for (index, value) in self.state.items():
        counts[index][value] += 1
    for node in self.net.nodes_by_index:
      for value in node.support:
        counts[node][value] /= num_transitions
    return counts

  def transition(self):
    raise NotImplementedError()    


class RejectionChain(MarkovChain):
  """Rejection sampler."""

  def __init__(self, net, rng, evidence):
    super(RejectionChain, self).__init__(net, rng)
    self.evidence = evidence

  def initialize_state(self):
    self.transition()

  def transition(self):
    accepted = False
    world = None
    while not accepted:
      world = self.net.sample(random_world.RandomWorld())
      accepted = True
      for (node, value) in self.evidence.items():
        if world[node] != value:
          accepted = False
          break
    self.state = world


class GibbsChain(MarkovChain):
  """Gibbs sampler."""

  def __init__(self, net, rng, evidence):
    """Initialize Gibbs sampler.

    Args:
      evidence: a mapping from nodes to values
    """
    super(GibbsChain, self).__init__(net, rng)
    self.evidence = evidence
    self.gibbs_distributions = {}
    for node in self.net.nodes():
      self.gibbs_distributions[node] = gibbs.all_gibbs_distributions(node, rng)

  def initialize_state(self):
    """Initialize from prior, set evidence nodes."""
    accepted = False
    while not accepted:
      self.state = self.net.sample(self.evidence)
      accepted = self.net.log_probability(self.state) != utils.LOG_PROB_0

  def transition(self):
    for node in self.net.nodes():
      if node not in self.evidence:
        self.update_node(node)

  def update_node(self, node):
    markov_blanket_vals = tuple(
      [self.state.data[var.index] for var in node.markov_blanket])
    gibbs_dist = self.gibbs_distributions[node][markov_blanket_vals]
    self.state.data[node.index] = gibbs_dist.sample()
    
