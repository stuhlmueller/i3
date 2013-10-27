from __future__ import division

import collections
import math

from i3 import dist
from i3 import utils

from scipy import stats
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors


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
      self.learner.fit([o[0] for o in observations], [o[1] for o in observations])

  def sample(self, params):
    # TODO track error variance
    return self.learner.predict(params)

  def log_probability(self, params, value):
    # TODO track error variance
    pred = self.learner.predict(params)
    return -(pred-value)**2


class KnnGaussianLearner(dist.ContinuousDistribution):

  def __init__(self, rng, k):
    super(LinRegLearner, self).__init__(rng)
    self.k = k
    self.observations = []
    self.nn = None

  def observe(self, params, value):
    self.observations.append((params, value))
    self.nn = None

  def finalize(self):
    if self.nn is None:
        xs = [x for (x, _) in self.observations if x]
        if not xs:
            return [], []
        self.nn = NearestNeighbors()
        self.nn.fit(xs)

  def get_knns(self, params):
    distance_array, index_array = self.nn.kneighbors(
            params, min(self.k, len(self.pairs)), return_distance=True)
    distances = list(distance_array[0])
    indices = list(index_array[0])
    elements = [self.pairs[i] for i in indices]
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


