"""Markov chain inference for Bayesian networks."""

from i3 import utils
from i3 import gibbs


class MarkovChain(object):

  def __init__(self, net, rng):
    self.net = net
    self.rng = rng

  def initialize_state(self):
    raise NotImplementedError()

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
    while not accepted:
      random_world = self.net.sample({})
      accepted = True
      for (node, value) in self.evidence.items():
        if random_world[node] != value:
          accepted = False
          break
    self.state = random_world


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
      self.gibbs_distributions[node] = gibbs.all_gibbs_distributions(
        node, node.support(), rng)

  def initialize_state(self):
    """Initialize from prior, set evidence nodes."""
    accepted = False
    while not accepted:
      self.state = self.net.sample(self.evidence)
      accepted = self.net.log_probability(self.state) != utils.LOG_PROB_0

  def transition(self):
    for node in self.rng.random_permutation(self.net.nodes()):
      if node not in self.evidence:
        self.update_node(node)

  def update_node(self, node):
    self.markov_blanket_vars = sorted(node.markov_blanket())
    markov_blanket_vals = tuple([self.state[var] for var in self.markov_blanket_vars])
    gibbs_dist = self.gibbs_distributions[node][markov_blanket_vals]
    self.state[node] = gibbs_dist.sample()
    
