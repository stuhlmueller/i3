"""Tests for Bayes net and Bayes net nodes."""

import collections

from i3 import utils
from i3.networks import binary_net
from i3.networks import sprinkler_net


class TestBinaryBayesNet(object):
  """Test a two-node Bayes net."""

  def setup(self):
    """Set up random stream and network."""
    rng = utils.RandomState(seed=0)
    self.net = binary_net.get(rng)
    self.node_1 = self.net.find_node("node_1")
    self.node_2 = self.net.find_node("node_2")
  
  def test_node(self):
    """Test a single BayesNetNode."""
    assert self.node_1.sample({}) == True
    assert self.node_1.log_probability({}, True) == utils.LOG_PROB_1
    assert self.node_1.log_probability({}, False) == utils.LOG_PROB_0
    assert self.node_1.name == str(self.node_1) == "node_1"

  def test_markov_blanket(self):
    """Check that Markov blanket is correct."""
    assert self.node_1.markov_blanket() == set([self.node_2])
    assert self.node_2.markov_blanket() == set([self.node_1])
  
  def test_network(self):
    """Test a simple Bayesian network."""
    # First random world (prior)
    random_world = self.net.sample()
    assert random_world[self.node_1] == True
    assert random_world[self.node_2] == False
    assert self.net.log_probability(random_world) == utils.LOG_PROB_1
    # Second random world (node 1 fixed)
    random_world = self.net.sample({self.node_1: False})
    assert random_world[self.node_2] == True
    assert self.net.log_probability(random_world) == utils.LOG_PROB_0


class TestSprinklerBayesNet(object):
  """Test a three-node rain/sprinkler/grass network."""

  def setup(self):
    """Set up random stream and network."""    
    rng = utils.RandomState(seed=0)
    self.net = sprinkler_net.get(rng)

  def test_network(self):
    n = 10000
    counts = collections.defaultdict(lambda: 0)
    for _ in xrange(n):
      random_world = self.net.sample()
      for (node, value) in random_world.items():
        if value:
          counts[node.name] += 1
    assert 1800 < counts["Rain"] < 2200
    assert 3000 < counts["Sprinkler"] < 3400
    assert 4200 < counts["Grass"] < 4600
