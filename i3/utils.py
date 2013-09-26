"""Utilities for stochastic sampling and probability calculations."""
from __future__ import division

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


def logsumexp(a):
  """Compute log of sum of exponentials of array."""
  array = np.array(a)
  array_max = array.max(axis=0)
  out = np.log(np.sum(np.exp(array - array_max), axis=0))
  out += array_max
  return out


def normalize(array):
  """Divide array by its sum to make it sum to 1."""
  Z = np.sum(array)
  return [i / Z for i in array]


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


def assert_in_interval(k, p, n, confidence=.95):
  """Check that observed number of heads is in confidence interval.
  
  Args:
    k: observed count of heads
    p: true probability of heads
    n: number of coin flips
    confidence: the probability mass that we want to cover

  FIXME: Take a more principled approach.
  """
  k_min, k_max = n * np.array(stats.beta(p * n, (1-p) * n).interval(confidence))
  assert k_min <= k <= k_max


def is_sorted(lst):
  """Return True if list is sorted, False otherwise."""
  return all(lst[i] <= lst[i+1] for i in xrange(len(lst)-1))


class RandomState(np.random.RandomState):
  """Extend numpy's RandomState with more sampling functions."""

  def categorical_sampler(self, values, probabilities):
    """Return a categorical sampler for given values and probabilities."""
    bins = np.add.accumulate(probabilities)
    def sampler():
      index = np.digitize([self.rand()], bins)[0]
      return values[index]
    return sampler

  def random_permutation(self, obj):
    """Return permuted copy of array. If given int, create range array."""
    if isinstance(obj, (int, np.integer)):
      array = np.arange(obj)
    else:
      array = np.array(obj)
    self.shuffle(array)
    return array
