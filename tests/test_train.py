from __future__ import division

import math
import pytest

from i3 import exact_inference
from i3 import invert
from i3 import mcmc
from i3 import random_world
from i3 import train
from i3 import utils
from i3.networks import sprinkler_net
from i3.networks import triangle_net


class TestSprinklerBayesNet(object):
  """Test training on a three-node rain/sprinkler/grass network."""

  def setup(self):
    self.rng = utils.RandomState(seed=0)
    self.net = sprinkler_net.get(self.rng)

  @pytest.mark.slow
  @pytest.mark.parametrize(
    "evidence_index,precompute_gibbs",
    utils.lexicographic_combinations([[0, 1, 2, 3], [True, False]]))
  def test(self, evidence_index, precompute_gibbs):
    evidence = sprinkler_net.evidence(evidence_index)
    evidence_nodes = [self.net.nodes_by_index[index]
                      for index in evidence.keys()]
    inverse_map = invert.compute_inverse_map(
      self.net, evidence_nodes, self.rng)
    assert len(inverse_map) == 2
    trainer = train.Trainer(self.net, inverse_map, precompute_gibbs)
    num_samples = 30000
    for _ in xrange(num_samples):
      sample = self.net.sample()
      trainer.observe(sample)
    trainer.finalize()

    # Compare marginal log probability for evidence node with prior marginals.
    empty_world = random_world.RandomWorld()
    enumerator = exact_inference.Enumerator(self.net, empty_world)
    exact_marginals = enumerator.marginals()
    for evidence_node in evidence_nodes:
      for value in [0, 1]:
        log_prob_true = math.log(exact_marginals[evidence_node.index][value])
        for inverse_net in inverse_map.values():
          log_prob_empirical = inverse_net.nodes_by_index[
            evidence_node.index].log_probability(empty_world, value)
          print abs(log_prob_true - log_prob_empirical)
          assert abs(log_prob_true - log_prob_empirical) < .02

    # For each inverse network, take unconditional samples, compare
    # marginals to prior network.
    num_samples = 30000
    for inverse_net in inverse_map.values():
      counts = [[0, 0], [0, 0], [0, 0]]
      for _ in xrange(num_samples):
        world = inverse_net.sample()
        for (index, value) in world.items():
          counts[index][value] += 1
      for index in [0, 1, 2]:
        true_dist = enumerator.marginalize_node(
          self.net.nodes_by_index[index])
        empirical_dist = utils.normalize(counts[index])
        for (p_true, p_empirical) in zip(true_dist, empirical_dist):
          print abs(p_true - p_empirical)
          assert abs(p_true - p_empirical) < .02


class TestUAIBayesNet(object):
  """Test training on triangle network."""

  def setup(self):
    self.rng = utils.RandomState(seed=0)
    self.net = triangle_net.get(self.rng)
    self.evidence = triangle_net.evidence(0)

  def train_from_prior(self, trainers, num_samples):
    for _ in xrange(num_samples):
      sample = self.net.sample()
      for trainer in trainers:
        trainer.observe(sample)

  def train_from_gibbs(self, trainers, num_samples, world):
    gibbs_sampler = mcmc.GibbsChain(
      self.net, self.rng, world)
    gibbs_sampler.initialize_state()
    for _ in xrange(num_samples):
      gibbs_sampler.transition()
      for trainer in trainers:
        trainer.observe(gibbs_sampler.state)    

  def train_from_gibbs_prior(self, trainers, num_samples):
    world = random_world.RandomWorld()
    self.train_from_gibbs(trainers, num_samples, world)

  def train_from_gibbs_posterior(self, trainers, num_samples):
    self.train_from_gibbs(trainers, num_samples, self.evidence)

  @pytest.mark.slow
  @pytest.mark.parametrize(
    "training_source", ["prior", "gibbs-prior", "gibbs-posterior"])
  def test(self, training_source):
    """Compare dist on final nodes with and without Gibbs precomputation."""
    evidence_nodes = [self.net.nodes_by_index[index]
                      for index in self.evidence.keys()]

    # Set up two trainers, one with Gibbs precomputation, one without.
    print "Computing inverse maps..."
    inverse_map_with_gibbs = invert.compute_inverse_map(
      self.net, evidence_nodes, self.rng, max_inverse_size=1)
    inverse_map_without_gibbs = invert.compute_inverse_map(
      self.net, evidence_nodes, self.rng, max_inverse_size=1)
    trainer_with_gibbs = train.Trainer(
      self.net, inverse_map_with_gibbs, precompute_gibbs=True)
    trainer_without_gibbs = train.Trainer(
      self.net, inverse_map_without_gibbs, precompute_gibbs=False)
    trainers = [trainer_with_gibbs, trainer_without_gibbs]

    # Train based on sampled data.
    print "Training..."
    num_samples = 50000
    if training_source == "prior":
      self.train_from_prior(trainers, num_samples)
    elif training_source == "gibbs-prior":
      self.train_from_gibbs_prior(trainers, num_samples)
    elif training_source == "gibbs-posterior":
      self.train_from_gibbs_posterior(trainers, num_samples)      
    else:
      raise ValueError("Unknown training source.")

    # Go through all nodes, check that all estimated conditional
    # distributions are close to true distributions.
    print "Comparing distributions..."
    error = 0.0
    num_checks = 0
    for node in inverse_map_with_gibbs.keys():
      net_with_gibbs = inverse_map_with_gibbs.get_net(node)
      net_without_gibbs = inverse_map_without_gibbs.get_net(node)
      gibbs_dist = net_with_gibbs.nodes_by_index[node.index].distribution
      estimated_dist = net_without_gibbs.nodes_by_index[node.index].distribution
      for markov_blanket_values in utils.lexicographic_combinations(
          [[0, 1]] * len(node.markov_blanket)):
        for value in node.support:
          true_value = math.exp(
            gibbs_dist.log_probability(markov_blanket_values, value))
          estimated_value = math.exp(
            estimated_dist.log_probability(markov_blanket_values, value))
          error += abs(true_value - estimated_value)
          num_checks += 1
    average_error = error / num_checks
    print average_error
    assert average_error < .05
