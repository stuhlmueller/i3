from __future__ import division

import collections
import math
from sklearn import linear_model

from i3 import dist
from i3 import gibbs
from i3 import utils


class CountLearner(dist.DiscreteDistribution):
  """Learn a family of distributions by counting."""

  def __init__(self, support, rng):
    super(CountLearner, self).__init__(rng)
    self._support = support
    self.counts = collections.defaultdict(
      lambda: [1] * len(support))

  def log_probability(self, params, value):
    """Return probability of value given values indicating family."""
    counts = self.counts[tuple(params)]
    probability = counts[value] / sum(counts)
    return math.log(probability)

  def observe(self, params, value):
    """Increment count of value for chosen family."""
    self.counts[tuple(params)][value] += 1

  def sample(self, params):
    """Sample from family indicated by params."""
    probabilities = utils.normalize(self.counts[tuple(params)])
    sampler = self.rng.categorical_sampler(self.support(params), probabilities)
    return sampler()

  def support(self, params):
    """Return values in support of learner."""
    return self._support

  def finalize(self):
    """No compilation step necessary."""
    pass


class GibbsLearner(dist.DiscreteDistribution):
  """Learn a family of distributions by exact computation of conditionals."""

  def __init__(self, node, rng):
    super(GibbsLearner, self).__init__(rng)
    self.gibbs_distributions = gibbs.all_gibbs_distributions(node, rng)

  def log_probability(self, params, value):
    return self.gibbs_distributions[tuple(params)].log_probability(None, value)

  def observe(self, params, value):
    # Gibbs learner doesn't make use of observations.
    pass

  def sample(self, params):
    return self.gibbs_distributions[tuple(params)].sample(None)

  def support(self, params):
    return self.gibbs_distributions[tuple(params)].support(None)

  def finalize(self):
    pass


identity_transformer = lambda xs: xs

square_transformer = lambda xs: [xi*xj for xi in xs for xj in xs]

class LogisticRegressionLearner(dist.DiscreteDistribution):
  """Learn a family of distributions using (batch) logistic regression."""
  
  def __init__(self, support, rng, transform_inputs=None):
    super(LogisticRegressionLearner, self).__init__(rng)
    self.predictor = linear_model.LogisticRegression(penalty="l2")
    self.inputs = []
    self.outputs = []
    self._support = support
    if transform_inputs is None:
      self.transform_inputs = identity_transformer
    else:
      self.transform_inputs = transform_inputs

  def log_probability(self, params, value):
    inputs = self.transform_inputs(params)
    probs = self.predictor.predict_proba(inputs)[0]
    return probs[value]

  def observe(self, params, value):
    inputs = self.transform_inputs(params)
    self.inputs.append(inputs)
    self.outputs.append(value)

  def sample(self, params):
    inputs = self.transform_inputs(params)    
    probs = self.predictor.predict_proba(inputs)[0]
    return self.rng.categorical(self._support, probs)

  def support(self, params):
    return self._support

  def finalize(self):
    self.predictor.fit(self.inputs, self.outputs)
    self.inputs = []
    self.outputs = []
    assert list(self.predictor.classes_) == self._support
    assert utils.is_range(self._support)
