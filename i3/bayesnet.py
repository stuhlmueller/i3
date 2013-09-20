"""Data structures for Bayes nets and nodes."""

import copy
import networkx

from i3 import distribution
from i3 import random_world


class BayesNetNode(object):
  """A single node in a Bayesian network."""

  def __init__(self, name, get_distribution, full_support=None, net=None):
    """Initialize Bayes net node based on sampling/scoring functions, support.

    Args:
      name: a string
      get_distribution: maps parent values to distribution
      full_support: list of values. if given, allows looking up node
        support without random world
    """
    self.name = name
    self.get_distribution = get_distribution
    self.full_support = full_support
    self.net = net

  def __str__(self):
    return str(self.name)

  def __repr__(self):
    return str(self.name)

  def __cmp__(self, other):
    return cmp(self.name, other.name)
    
  def set_net(self, net):
    """Set the Bayes net associated with this node (once)."""
    assert not self.net
    self.net = net

  def parents(self):
    """Return sorted list of children (BayesNetNodes)."""
    assert self.net    
    return sorted(self.net.predecessors(self))

  def children(self):
    """Return sorted list of parents (BayesNetNodes)."""
    assert self.net
    return sorted(self.net.successors(self))

  def parent_values(self, world):
    """Extract list of parent values from random world."""
    return [world[parent] for parent in self.parents()]

  def support(self, world=None):
    """Return supported values of node given random world."""
    assert world or self.full_support
    if world:
      return self.distribution(world).support()
    else:
      return self.full_support

  def distribution(self, world):
    """Return distribution of node conditioned on parents."""
    return self.get_distribution(*self.parent_values(world))

  def markov_blanket(self):
    """Return set of nodes in the Markov blanket of this node."""
    coparents = [parent for child in self.children() for parent in child.parents()]
    blanket = list(self.parents()) + list(self.children()) + coparents
    return set(node for node in blanket if node != self)
        
  def sample(self, world):
    """Sample a value for this node given parent node values.

    Args:
      world: a random world

    Returns:
      a sampled value
    """
    return self.distribution(world).sample()

  def log_probability(self, world, node_value):
    """Return the log probability of node_value for this node given context.

    Args:
      world: a random world
      node_value: a value for this node

    Returns:
      score: a log probability
    """
    return self.distribution(world).log_probability(node_value)


class CategoricalNode(BayesNetNode):
  """A BayesNetNode for categorical distributions."""
  def __init__(self, name, values, get_probabilities, rng, net=None):
    get_distribution = (
      lambda *parent_values:
      distribution.CategoricalDistribution(
        values, get_probabilities(*parent_values), rng))
    super(CategoricalNode, self).__init__(
      name=name,
      get_distribution=get_distribution,
      full_support=values)


class BayesNet(networkx.DiGraph):
  """A Bayesian network."""

  def __init__(self, name, nodes, edges, **attr):
    """Initializes Bayesian network.

    Args:
      name: a string
      nodes: a list of BayesNetNodes
    """
    super(BayesNet, self).__init__(**attr)
    self.add_nodes_from(nodes)
    self.add_edges_from(edges)
    self.sorted_nodes = tuple(networkx.topological_sort(self))
    for node in nodes:
      node.set_net(self)
  
  def sample(self, world=None):
    """Sample an assignment to all nodes in the network.

    Args:
      world: a random world

    Returns:
      a new random world
    """
    if world:
      world = world.copy()
    else:
      world = random_world.RandomWorld()
    for node in self.sorted_nodes:
      if not node in world:
        world[node] = node.sample(world)
    return world

  def log_probability(self, world):
    """Return the total log probability of the given random world.

    Args:
      world: a random world

    Returns:
      log probability
    """
    assert len(world) == self.number_of_nodes()
    log_prob = 0.0
    for node in world:
      log_prob += node.log_probability(world, world[node])
    return log_prob

  def find_node(self, name):
    """Return the (first) node with the given name, or fail.

    Args:
      name: a string

    Returns:
      a BayesNetNode
    """
    for node in self.sorted_nodes:
      if node.name == name:
        return node
    raise Exception("Node %s not found!", name)
