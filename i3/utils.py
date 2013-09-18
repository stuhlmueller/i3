#!/usr/bin/env python

"""Utilities for stochastic sampling and probability calculations."""

import math
import numpy as np


NEGATIVE_INFINITY = float('-inf')

LOG_PROB_0 = NEGATIVE_INFINITY

LOG_PROB_1 = 0.0


def safe_log(num):
  """Like math.log, but returns -infinity on 0."""
  if num == 0.0:
    return NEGATIVE_INFINITY
  return math.log(num)


class RandomState(np.random.RandomState):
  """Extend numpy's RandomState with more sampling functions."""

  def categorical_sampler(self, values, probabilities):
    """Return a categorical sampler for given values and probabilities."""
    bins = np.add.accumulate(probabilities)
    def sampler():
      index = np.digitize([self.rand()], bins)[0]
      return values[index]
    return sampler
