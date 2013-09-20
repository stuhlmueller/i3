"""Utilities for stochastic sampling and probability calculations."""
from __future__ import division

import itertools
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
    

def all_random_worlds(variable_names, support):
  """Return iterable over all possible random worlds.

  Args:
    variable_names: a list of variable names
    support: a list of possible values

  Returns:
    iterable of dictionaries mapping variables to values
  """
  for values in itertools.product(*[support]*len(variable_names)):
    yield dict(zip(variable_names, values))
