from __future__ import division

import collections
import math

from i3 import dist
from i3 import gibbs
from i3 import utils

from scipy import stats
from sklearn import linear_model
from sklearn import neighbors


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
    return utils.safe_log(probability)

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


class LinRegLearner(dist.ContinuousDistribution):

  def __init__(self, rng):
    super(LinRegLearner, self).__init__(rng)
    self.observations = []
    self.learner = None

  def observe(self, params, value):
    self.observations.append((params, value))
    self.learner = None

  def finalize(self):
    if self.learner is None:
      self.learner = linear_model.LinearRegression()
      xs = [o[0] for o in self.observations]
      ys = [o[1] for o in self.observations]
      self.learner.fit(xs, ys)

  def sample(self, params):
    # TODO track error variance
    return self.learner.predict(params)

  def log_probability(self, params, value):
    # TODO track error variance
    pred = self.learner.predict(params)
    return -(pred - value) ** 2


class KnnGaussianLearner(dist.ContinuousDistribution):

  def __init__(self, rng, k):
    super(KnnGaussianLearner, self).__init__(rng)
    self.k = k
    self.observations = []
    self.nn = None

  def observe(self, params, value):
    self.observations.append((params, value))
    self.nn = None

  def finalize(self):
    if self.nn is None:
      if not self.observations:
        raise Exception('no observations')
      if not self.observations[0][0]:
        # 0-length vectors are not allowed
        xs = [[0] for _ in self.observations]
      else:
        xs = [x for (x, _) in self.observations]
      self.nn = neighbors.NearestNeighbors()
      self.nn.fit(xs)

  def get_knns(self, params):
    self.finalize()
    distance_array, index_array = self.nn.kneighbors(
      params, min(self.k, len(self.observations)), return_distance=True)
    distances = list(distance_array[0])
    indices = list(index_array[0])
    elements = [self.observations[i][1] for i in indices]
    return elements, distances

  def get_density_estimator(self, params):
    (knns, dists) = self.get_knns(params)
    return stats.gaussian_kde(knns)

  def sample(self, params):
    kde = self.get_density_estimator(params)
    return kde.resample(size=1)[0][0]

  def log_probability(self, params, value):
    kde = self.get_density_estimator(params)
    p = kde.evaluate(value)
    assert p != 0.0
    return math.log(p)


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

square_transformer = lambda xs: [xi * xj for xi in xs for xj in xs]


class LogisticRegressionLearner(dist.DiscreteDistribution):
  # Only binary values supported for now.
  
  def __init__(self, support, rng, transform_inputs=None):
    super(LogisticRegressionLearner, self).__init__(rng)
    assert support == [0, 1]
    self.weights = None
    self.rate = 1.0
    self.n = 1
    if transform_inputs is None:
      self.transformer = identity_transformer
    else:
      self.transformer = transform_inputs

  def params_to_inputs(self, params):
    return tuple(self.transformer(params)) + (1,)

  def probability(self, inputs, value):
    logit = sum(theta_i * x_i for (theta_i, x_i) in zip(self.weights, inputs))
    p_on = 1.0 / (1.0 + math.exp(-logit))
    return value * p_on + (1 - value) * (1 - p_on)

  def log_probability(self, params, value):
    assert value in (0, 1)
    inputs = self.params_to_inputs(params)
    assert len(inputs) == len(self.weights)
    return utils.safe_log(self.probability(inputs, value))

  def observe(self, params, value):
    assert value in (0, 1)
    inputs = self.params_to_inputs(params)    
    if self.weights is None:
      self.weights = [0.0] * len(inputs)
    else:
      assert len(self.weights) == len(inputs)
    p_on = self.probability(inputs, 1)
    for i, weight in enumerate(self.weights):
      self.weights[i] += (self.rate / math.sqrt(self.n)) * (value - p_on) * inputs[i]
    self.n += 1
  
  def sample(self, params):
    inputs = self.params_to_inputs(params)
    assert len(inputs) == len(self.weights)    
    p_on = self.probability(inputs, 1)
    return 1 if self.rng.flip(p_on) else 0

  def support(self, params):
    return [0, 1]

  def finalize(self):
    pass
