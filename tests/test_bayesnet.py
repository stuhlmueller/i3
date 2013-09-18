"""Tests for Bayes net and Bayes net nodes."""

from i3 import bayesnet
from i3 import distribution
from i3 import utils


class TestBayesNet(object):
  """Test a two-node Bayes net."""

  def setup(self):
    """Set up two nodes and network."""
    rng = utils.RandomState(seed=0)
    get_dist_1 = lambda: distribution.CategoricalDistribution(
      [True], [1.0], rng)
    self.node_1 = bayesnet.BayesNetNode("node_1", [], get_dist_1)
    get_dist_2 = lambda parent_value: distribution.CategoricalDistribution(
      [not(parent_value)], [1.0], rng)
    self.node_2 = bayesnet.BayesNetNode("node_2", [self.node_1], get_dist_2)
    self.node_1.add_child(self.node_2)
    self.net = bayesnet.BayesNet([self.node_1, self.node_2])
  
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
