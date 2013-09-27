"""Tests for Bayes net and Bayes net nodes."""

import collections
import numpy as np
import math
import pytest

from i3 import random_world
from i3 import utils
from i3.networks import binary_net
from i3.networks import sprinkler_net


class TestBinaryBayesNet(object):
  """Test a two-node Bayes net."""

  def setup(self):
    """Set up random stream and network."""
    rng = utils.RandomState(seed=0)
    self.net = binary_net.get(rng)
    self.node_1 = self.net.nodes_by_index[0]
    self.node_2 = self.net.nodes_by_index[1]
  
  def test_nodes(self):
    """Test a single BayesNetNode."""
    world = random_world.RandomWorld(2)
    assert self.node_1.sample(world) == 1
    assert self.node_1.log_probability(world, 0) == utils.LOG_PROB_0    
    assert self.node_1.log_probability(world, 1) == utils.LOG_PROB_1

  def test_markov_blanket(self):
    """Check that Markov blanket is correct."""
    assert self.node_1.markov_blanket == [self.node_2]
    assert self.node_2.markov_blanket == [self.node_1]
  
  def test_network(self):
    """Test a simple Bayesian network."""
    # First random world (prior)
    world = self.net.sample()
    assert world[self.node_1] == 1
    assert world[self.node_2] == 2
    assert self.net.log_probability(world) == utils.LOG_PROB_1
    # Second random world (node 1 fixed)
    given_world = random_world.RandomWorld(2)
    given_world[self.node_1] = 0
    world = self.net.sample(given_world)
    assert world[self.node_2] == 1
    assert self.net.log_probability(world) == utils.LOG_PROB_0

  def test_representation(self):
    """Check that calling string methods works."""
    assert str(self.net) == repr(self.net)


class TestSprinklerBayesNet(object):
  """Test a three-node rain/sprinkler/grass network."""

  def setup(self):
    """Set up random stream and network."""    
    rng = utils.RandomState(seed=0)
    self.net = sprinkler_net.get(rng)
    self.rain_node = self.net.find_node("Rain")
    self.sprinkler_node = self.net.find_node("Sprinkler")
    self.grass_node = self.net.find_node("Grass")
    self.n = 10000

  def test_nodes(self):
    """Check sampling and probability functions of nodes."""
    world = random_world.RandomWorld(3)
    world[self.rain_node] = 1
    np.testing.assert_almost_equal(
      math.exp(self.sprinkler_node.log_probability(world, 1)),
      0.4)
    sprinkler_count = 0
    for _ in xrange(self.n):
      value = self.sprinkler_node.sample(world)
      if value == 1:
        sprinkler_count += 1
    utils.assert_in_interval(sprinkler_count, .4, self.n, .95)
    world = random_world.RandomWorld(3)
    world[self.rain_node] = 0
    world[self.sprinkler_node] = 1
    np.testing.assert_almost_equal(
      math.exp(self.grass_node.log_probability(world, 1)),
      0.7)

  def test_topological_order(self):
    """Check that nodes are sorted in topological order."""
    assert self.net.nodes_by_topology == (
      self.rain_node,
      self.sprinkler_node,
      self.grass_node)

  def test_network(self):
    """Check that marginals are correct when sampling from prior."""
    counts = collections.defaultdict(lambda: 0)
    for _ in xrange(self.n):
      world = self.net.sample()
      for (i, value) in enumerate(world):
        assert value in [0, 1]
        if value == 1:
          counts[self.net.nodes_by_index[i].name] += 1
    utils.assert_in_interval(counts["Rain"], .2, self.n, .95)
    utils.assert_in_interval(counts["Sprinkler"], .872, self.n, .95)
    utils.assert_in_interval(counts["Grass"], .7332, self.n, .95)        

  def test_find_node(self):
    """Check that finding nodes by name works."""
    with pytest.raises(ValueError):
      node = self.net.find_node("Foo")
