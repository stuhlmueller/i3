from __future__ import division

import math
import numpy as np

from i3 import learn
from i3 import utils


def check_sampler(sampler, probabilities, num_samples, confidence=.95):
  counts = [0] * len(probabilities)
  for _ in xrange(num_samples):
    sample = sampler()
    counts[sample] += 1
  for i, count in enumerate(counts):
    utils.assert_in_interval(count, probabilities[i], num_samples, confidence)


def check_scorer(log_scorer, probabilities):
  for i, probability in enumerate(probabilities):
    np.testing.assert_almost_equal(log_scorer(i), math.log(probability))


def test_count_learner():
  rng = utils.RandomState(seed=0)
  learner = learn.CountLearner(support=[0, 1], rng=rng)
  num_samples = 10000

  # Check predictions with empty input, no observations
  null = ()
  check_sampler(lambda: learner.sample(null), [0.5, 0.5], num_samples)
  check_scorer(lambda v: learner.log_probability(null, v), [0.5, 0.5])

  # Check predictions with empty input, observations
  learner.observe(null, 1)
  check_sampler(lambda: learner.sample(null), [1 / 3, 2 / 3], num_samples)
  check_scorer(lambda v: learner.log_probability(null, v), [1 / 3, 2 / 3])
  learner.observe(null, 1)
  check_sampler(lambda: learner.sample(null), [1 / 4, 3 / 4], num_samples)
  check_scorer(lambda v: learner.log_probability(null, v), [1 / 4, 3 / 4])

  # Check with different input
  inputs = (0, 1)
  learner.observe(inputs, 0)
  check_sampler(lambda: learner.sample(inputs), [2 / 3, 1 / 3], num_samples)
  check_scorer(lambda v: learner.log_probability(inputs, v), [2 / 3, 1 / 3])
  check_sampler(lambda: learner.sample(null), [1 / 4, 3 / 4], num_samples)
  check_scorer(lambda v: learner.log_probability(null, v), [1 / 4, 3 / 4])

  # For the sake of completeness
  learner.finalize()
