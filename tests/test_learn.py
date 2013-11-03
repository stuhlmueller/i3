from __future__ import division

import math
import numpy as np
import pytest

from i3 import exact_inference
from i3 import learn
from i3 import random_world
from i3 import utils
from i3.networks import sprinkler_net


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
  """Verify that count learner makes correct predictions."""
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


def test_gibbs_learner():
  """Verify that Gibbs learner makes same predictions as enumerator."""
  num_samples = 10000
  rng = utils.RandomState(seed=0)
  net = sprinkler_net.get(rng)
  for node in net.nodes_by_index:
    learner = learn.GibbsLearner(node, rng)
    learner.finalize()
    for markov_blanket_values in utils.lexicographic_combinations(
        [[0, 1]] * len(node.markov_blanket)):
      world = random_world.RandomWorld(
        [n.index for n in node.markov_blanket],
        markov_blanket_values)
      enumerator = exact_inference.Enumerator(net, world)
      probabilities = enumerator.marginalize_node(node)
      check_scorer(
        lambda v: learner.log_probability(
          markov_blanket_values, v), probabilities)
      check_sampler(
        lambda: learner.sample(
          markov_blanket_values), probabilities, num_samples)


input_transformers = [learn.identity_transformer, learn.square_transformer]

class TestLogisticRegressionLearner(object):

  @pytest.mark.parametrize("transform_inputs", input_transformers)
  def test_learner(self, transform_inputs):
    """Test learner on its own."""
    rng = utils.RandomState(seed=0)
    learner = learn.LogisticRegressionLearner(
      [0, 1], rng, transform_inputs=transform_inputs)
    training_data = [
      ((0, 0, 0), 0),
      ((0, 0, 1), 1),
      ((0, 1, 0), 1),
      ((1, 0, 0), 0),
      ((0, 0, 0), 0),
      ((0, 0, 1), 1),
      ((0, 1, 0), 1),
      ((1, 0, 0), 0)
    ]
    for (inputs, output) in training_data:
      learner.observe(inputs, output)
    learner.finalize()
    score = learner.log_probability
    assert score((0, 1, 1), 0) < score((0, 1, 1), 1)
    assert score((1, 1, 0), 0) > score((1, 1, 0), 1)

  @pytest.mark.parametrize("transform_inputs", input_transformers)
  def test_learner_accuracy(self, transform_inputs):
    """Test that learner gets relative probabilities right"""
    rng = utils.RandomState(seed=0)
    learner = learn.LogisticRegressionLearner(
      [0, 1], rng, transform_inputs=transform_inputs)
    training_data = [0] * 666 + [1] * 333
    for datum in training_data:
      learner.observe((), datum)
    learner.finalize()
    print math.exp(learner.log_probability((), 0))
    print math.exp(learner.log_probability((), 1))
    assert math.log(0.656) < learner.log_probability((), 0) < math.log(0.676)
    assert math.log(0.323) < learner.log_probability((), 1) < math.log(0.343)

  @pytest.mark.parametrize("transform_inputs", input_transformers)    
  def test_predict_proba(self, transform_inputs):
    """Test incomplete label data"""
    rng = utils.RandomState(seed=0)
    learner = learn.LogisticRegressionLearner(
      [0, 1, 2], rng, transform_inputs=transform_inputs)    
    training_data = [
      ((0, 0, 0), 0),
      ((1, 0, 0), 0),
      ((1, 1, 1), 1)
    ]
    for (inputs, output) in training_data:
      learner.observe(inputs, output)
    learner.finalize()
    score = learner.log_probability
    assert score((0, 0, 0), 0) > score((0, 0, 0), 1)
    assert score((1, 0, 0), 0) > score((1, 0, 0), 1)
    assert score((1, 1, 1), 1) > score((1, 1, 1), 0)    
