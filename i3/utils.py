"""Utilities for stochastic sampling and probability calculations."""

import math
import numpy as np
from scipy import stats


NEGATIVE_INFINITY = float('-inf')

LOG_PROB_0 = NEGATIVE_INFINITY

LOG_PROB_1 = 0.0


def safe_log(num):
  """Like math.log, but returns -infinity on 0."""
  if num == 0.0:
    return NEGATIVE_INFINITY
  return math.log(num)


def significantly_greater(a, b, alpha=0.05):
  """Perform one-sided t-test with null-hypothesis a <= b.

  Args:
    a: array
    b: array
    alpha: significance threshold

  Returns:
    True if null hypothesis rejected, false otherwise
  """
  t, probability = stats.ttest_ind(a, b)
  p_value = probability/2
  return p_value < alpha and t > 0  


class RandomState(np.random.RandomState):
  """Extend numpy's RandomState with more sampling functions."""

  def categorical_sampler(self, values, probabilities):
    """Return a categorical sampler for given values and probabilities."""
    bins = np.add.accumulate(probabilities)
    def sampler():
      index = np.digitize([self.rand()], bins)[0]
      return values[index]
    return sampler
