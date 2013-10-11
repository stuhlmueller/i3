from __future__ import division

import collections
import math

from i3 import utils


class CountLearner(object):

  def __init__(self, support, rng):
    self.support = support
    self.counts = collections.defaultdict(
      lambda: [1] * len(support))
    self.rng = rng

  def log_probability(self, inputs, output):
    counts = self.counts[tuple(inputs)]
    probability = counts[output] / sum(counts)
    return math.log(probability)

  def observe(self, inputs, output):
    self.counts[tuple(inputs)][output] += 1

  def sample(self, inputs):
    probabilities = utils.normalize(self.counts[tuple(inputs)])
    sampler = self.rng.categorical_sampler(self.support, probabilities)
    return sampler()

  def finalize(self):
    pass
