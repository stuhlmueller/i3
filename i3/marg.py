"""A random world subclass for storing marginals."""
from __future__ import division

from i3 import random_world


class Marginals(random_world.RandomWorld):
  """A mapping from BayesNetNodes to lists of probabilities."""

  def __sub__(self, other):
    assert len(self) == len(other)
    diff = Marginals()
    for i, ps in self.items():
      diff[i] = 0
      for j, p in enumerate(ps):
        q = other[i][j]
        diff[i] += abs(p-q)/len(ps)
    return diff
