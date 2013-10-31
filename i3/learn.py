from __future__ import division

import collections
import itertools
import math
import numpy
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
    assert utils.is_range(self._support)    
    if transform_inputs is None:
      self.transform_inputs = identity_transformer
    else:
      self.transform_inputs = transform_inputs
    self.classes_trained = None
    self.classes_not_trained = None    

  def log_probability(self, params, value):
    inputs = self.transform_inputs(params)
    probs = self.predict_probabilities(inputs)
    prob = probs[value]
    return utils.safe_log(prob)

  def observe(self, params, value):
    inputs = self.transform_inputs(params)
    self.inputs.append(inputs)
    self.outputs.append(value)

  def sample(self, params):
    inputs = self.transform_inputs(params)    
    probs = self.predict_probabilities(inputs)
    return self.rng.categorical(self._support, probs)

  def support(self, params):
    return self._support

  def finalize(self):
    self.predictor.fit(self.inputs, self.outputs)
    self.inputs = []
    self.outputs = []    
    try:
      self.classes_trained = self.predictor.classes_
    except AttributeError:
      # For older versions of sklearn.
      self.classes_trained = self.predictor.label_
    self.classes_not_trained = set(
      self.classes_trained).symmetric_difference(self._support)
    
  def predict_probabilities(self, inputs):
    # Wrapper around sklearn predict_proba that adds all classes
    probs = self.predictor.predict_proba(inputs)[0]
    prob_per_class = (zip(self.classes_trained, probs) +
                      zip(self.classes_not_trained, itertools.repeat(0.)))
    prob_per_class = sorted(prob_per_class)
    return [i[1] for i in prob_per_class]
