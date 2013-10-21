"""Markov chain inference for Bayesian networks."""
from __future__ import division

import math

from i3 import gibbs
from i3 import marg
from i3 import random_world
from i3 import utils


class MarkovChain(object):
  def __init__(self, rng):
    """Initialize Markov chain state."""
    self.rng = rng
    self.state = None

  def initialize_state(self):
    """Sample an initial Markov chain state."""
    raise NotImplementedError()

  def transition(self):
    """Transition to next Markov chain state."""
    raise NotImplementedError()


class BayesNetChain(MarkovChain):
  """A sequence of dependent (Bayes net) random variables."""

  def __init__(self, net, rng, evidence):
    """Initialize Markov chain state."""
    super(BayesNetChain, self).__init__(rng)
    self.net = net
    self.evidence = evidence

  def initialize_state(self):
    """Initialize from prior, set evidence nodes."""
    accepted = False
    while not accepted:
      self.state = self.net.sample(self.evidence)
      accepted = self.net.log_probability(self.state) != utils.LOG_PROB_0

  def marginals(self, num_transitions):
    """Compute marginal distribution of nodes.

    Compute marginal distribution of Bayes net nodes by repeatedly
    applying self transition kernel and storing state counts.

    Args:
      num_transitions: number of transitions to apply

    Returns:
      empirical marginals
    """
    counter = marg.MarginalCounter(self.net)
    for i in xrange(num_transitions):
      self.transition()
      counter.observe(self.state)
    return counter.marginals()


class RejectionChain(BayesNetChain):
  """Rejection sampler."""

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


class GibbsChain(BayesNetChain):
  """A Gibbs Markov chain."""

  def __init__(self, net, rng, evidence):
    """Initialize Gibbs sampler.

    Args:
      net: a BayesNet
      rng: a RandomState
      evidence: a mapping from nodes to values
    """
    super(GibbsChain, self).__init__(net, rng, evidence)
    self.gibbs_distributions = {}
    for node in self.net.nodes():
      self.gibbs_distributions[node] = gibbs.all_gibbs_distributions(node, rng)

  def transition(self):
    """Transition to next chain state by randomly updating each node in net."""
    for node in self.net.nodes():
      if node not in self.evidence:
        self.update_node(node)

  def update_node(self, node):
    """Update a single node using its conditional distribution."""
    markov_blanket_vals = tuple(
      [self.state.data[var.index] for var in node.markov_blanket])
    gibbs_dist = self.gibbs_distributions[node][markov_blanket_vals]
    self.state.data[node.index] = gibbs_dist.sample(None)


class InverseChain(BayesNetChain):
  """A blocked sampler (using a map of inverse networks)."""

  def __init__(self, net, bayesnet_map, rng, evidence, proposal_size):
    """Initialize MCMC chain with blocked resampling based on inverses.

    Args:
      net: a BayesNet
      bayesnet_map: a BayesNetMap
      rng: a RandomState
      evidence: a mapping from nodes to values
    """
    super(InverseChain, self).__init__(net, rng, evidence)
    self.evidence = evidence
    self.proposal_size = proposal_size
    self.inverse_nets = bayesnet_map.values()
    self.net = net

  def initialize_state(self):
    """Could use smarter inverse initializer here."""
    super(InverseChain, self).initialize_state()

  def transition(self):
    inverse_net = self.rng.choice(self.inverse_nets)

    # We never propose to evidence nodes
    proposal_nodes = [
      node for node in inverse_net.nodes_by_topology[-self.proposal_size:]
      if node.index not in self.evidence]

    proposal = self.state.copy()

    # New proposal, bw, fw, diff nodes
    logp_bwfw = utils.LOG_PROB_1
    diff_nodes = set()
    for node in proposal_nodes:
      proposal[node] = node.sample(proposal)
      logp_bwfw += node.log_probability(self.state, self.state[node])
      logp_bwfw -= node.log_probability(proposal, proposal[node])
      diff_nodes.add(self.net.nodes_by_index[node.index])
      for child in self.net.nodes_by_index[node.index].children:
        diff_nodes.add(child)

    # New/old state prob
    # The nodes here are nodes in the original Bayes net.
    logp_newold = utils.LOG_PROB_1
    for node in diff_nodes:
      logp_newold += node.log_probability(proposal, proposal[node])
      logp_newold -= node.log_probability(self.state, self.state[node])

    logp_acceptance = min(utils.LOG_PROB_1, logp_newold + logp_bwfw)

    accept = self.rng.flip(math.exp(logp_acceptance))

    if accept:
      self.state = proposal
