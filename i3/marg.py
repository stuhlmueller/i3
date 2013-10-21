"""A random world subclass for storing marginals."""
from __future__ import division

from i3 import random_world


class Marginals(random_world.RandomWorld):
  """A mapping from BayesNetNodes to lists of probabilities."""

  def __lt__(self, other):
    if type(other) == float:
      for v in self.values():
        if v >= other:
          return False
        return True
    else:
      return super(Marginals, self) < other

  def __gt__(self, other):
    if type(other) == float:
      for v in self.values():
        if v <= other:
          return False
        return True
    else:
      return super(Marginals, self) > other

  def __ge__(self, other):
    return not self < other

  def __le__(self, other):
    return not other < self

  def __sub__(self, other):
    assert len(self) == len(other)
    diff = Marginals()
    for i, ps in self.items():
      diff[i] = 0
      for j, p in enumerate(ps):
        q = other[i][j]
        diff[i] += abs(p - q) / len(ps)
    return diff


class MarginalCounter(object):
  """Compute marginals by observing samples."""

  def __init__(self, net):
    """Initialize counter using BayesNet."""
    self.net = net
    self.counts = self.get_empty_marginals()
    self.num_observations = 0

  def get_empty_marginals(self):
    """Return marginals that have entry 0 for all probabilities."""
    return Marginals(
      self.net.nodes_by_index,
      [[0] * len(node.support) for node in self.net.nodes_by_index]
    )    

  def observe(self, world):
    """Update counts from observed random world."""
    self.num_observations += 1
    for (index, value) in world.items():
      self.counts[index][value] += 1

  def marginals(self):
    """Return normalized marginals."""
    assert self.num_observations > 0
    margs = self.get_empty_marginals()
    for node in self.net.nodes_by_index:
      for value in node.support:
        margs[node][value] = self.counts[node][value] / self.num_observations
    return margs
