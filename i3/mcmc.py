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
