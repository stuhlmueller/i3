"""Probability distributions."""
from __future__ import division

import collections

from i3 import utils


class Distribution(object):
  """A probability distribution (sampler and scorer)."""

  def __init__(self, rng):
    self.rng = rng

  def sample(self, params):
    """Get a sample from the distribution."""
    raise NotImplementedError("sample")

  def log_probability(self, params, value):
    """Get the log probability of a value."""
    raise NotImplementedError("probability")


class DiscreteDistribution(Distribution):
  """A discrete probability distribution (sampler, scorer, and support)."""

  def support(self, params):
    raise NotImplementedError("support")


class CategoricalDistribution(DiscreteDistribution):
  """A distribution over a finite number of values."""

  def __init__(self, values, probabilities, rng):
    """Create a categorical distribution.

    Args:
      values: an iterable of associated values
      probabilities: an iterable of probabilites
    """
    super(CategoricalDistribution, self).__init__(rng)
    self.support_values = values
    self.sampler = None
    self.value_to_logprob = None
    self.probabilities = utils.normalize(probabilities)
    self.compile()

  def compile(self):
    self.sampler = self.rng.categorical_sampler(
      self.support_values, self.probabilities)
    self.value_to_logprob = collections.defaultdict(
      lambda: utils.LOG_PROB_0)
    for value, prob in zip(self.support_values, self.probabilities):
      self.value_to_logprob[value] = utils.safe_log(prob)

  def sample(self, params):
    """Sample a single value from the distribution."""
    assert not params
    return self.sampler()

  def log_probability(self, params, value):
    """Return the log probability of a given value."""
    return self.value_to_logprob[value]

  def support(self, params):
    assert not params
    return self.support_values
