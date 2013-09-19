"""Probability distributions."""
from __future__ import division

import collections

from i3 import utils


class Distribution(object):
  """A probability distribution (sampler and scorer)."""

  def __init__(self, rng):
    self.rng = rng

  def sample(self):
    """Get a sample from the distribution."""
    raise NotImplementedError('sample')

  def log_probability(self, value):
    """Get the log probability of a value."""
    raise NotImplementedError('probability')


class DiscreteDistribution(Distribution):
  """A discrete probability distribution (sampler, scorer, and support)."""

  def support(self):
    """Get a list of values in the support of this distribution."""
    raise NotImplementedError('support')


class CategoricalDistribution(DiscreteDistribution):
  """A distribution over a finite number of values."""

  def __init__(self, values, probabilities, rng):
    """Create a categorical distribution.

    Args:
      values: an iterable of associated values
      probabilities: an iterable of probabilites
    """
    super(CategoricalDistribution, self).__init__(rng)
    total = sum(probabilities)
    probabilities = [prob / total for prob in probabilities]
    self._sampler = rng.categorical_sampler(values, probabilities)
    self._value_to_logprob = collections.defaultdict(
      lambda: utils.LOG_PROB_0)
    self._support = []
    for value, prob in zip(values, probabilities):
      self._value_to_logprob[value] = utils.safe_log(prob)
      if prob != 0.0:
        self._support.append(value)

  def sample(self):
    """Sample a single value from the distribution."""
    return self._sampler()

  def log_probability(self, value):
    """Return the log probability of a given value."""
    return self._value_to_logprob[value]

  def support(self):
    """Return list of all values with non-zero probability."""
    return self._support
