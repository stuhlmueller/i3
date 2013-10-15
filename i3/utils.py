"""Utilities for stochastic sampling and probability calculations."""
from __future__ import division

import math
import numpy as np
from scipy import stats


NEGATIVE_INFINITY = float('-inf')

LOG_PROB_0 = NEGATIVE_INFINITY

LOG_PROB_1 = 0.0


def pop_n(stack, n):
  """Return and remove the first n elements from stack."""
  return [stack.popleft() for _ in xrange(n)]


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
  z = np.sum(array)
  return [i / z for i in array]


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
  p_value = probability / 2
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
  k_min, k_max = n * np.array(stats.beta(p * n, (1 - p) * n).interval(confidence))
  assert k_min <= k <= k_max


class RandomState(np.random.RandomState):
  """Extend numpy's RandomState with more sampling functions."""

  def categorical_sampler(self, values, probabilities):
    """Return a categorical sampler for given values and probabilities."""
    if not len(values) == len(probabilities):
      raise ValueError("Values and probabilities need to be of equal length!")
    if not values:
      raise ValueError("Categorical sampler needs at least one value!")
    bins = np.add.accumulate([0] + probabilities)

    def sampler():
      low = 0
      high = len(bins) - 1
      p = self.rand()
      while low < high - 1:
        mid = (low + high) // 2
        mid_cdf = bins[mid]
        if p < mid_cdf:
          high = mid
        else:
          low = mid
      return values[low]

    return sampler

  def flip(self, p):
    """Return True with probability p, False otherwise."""
    return self.rand() < p
  
  def random_permutation(self, obj):
    """Return permuted copy of array. If given int, create range array."""
    if isinstance(obj, (int, np.integer)):
      array = np.arange(obj)
    else:
      array = np.array(obj)
    self.shuffle(array)
    return array


def lexicographic_combinations(domains):
  """Returns lexicographically ordered combinations of values in domains.

  Args:
    domains: a list of lists [A, B, C, ...]

  Returns:
    a lexicographically ordered list of lists of values
    [[a0, b0, c0], [a0, b0, c1], ..., [an, bn, cn]]
  """
  if len(domains) == 1:
    for value in domains[0]:
      yield [value]
  else:
    for value in domains[0]:
      for lst in lexicographic_combinations(domains[1:]):
        yield [value] + lst


def reordered_list(old_order, new_order, old_list):
  """Given old and new ordering, return list with new ordering."""
  assert set(old_order) == set(new_order)
  assert len(old_order) == len(old_list)
  index_to_element = dict(zip(old_order, old_list))
  return [index_to_element[i] for i in new_order]
