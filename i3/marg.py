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
