"""Data structures for Bayes nets and nodes."""

import copy
import itertools
import math
import networkx

from i3 import distribution
from i3 import random_world


class BayesNetNode(object):
  """Categorical Bayes net node."""

  def __init__(self, index, domain_size=None, cpt_probabilities=None, name=None):
    """Initializes Bayes net node.

    Args:
      index: Bayesnet-unique identifier (integer)
      domain_size: number of elements in support
      cpt_probabilities: a list of cpt probabilities in UAI format
      name: a string (optional)
    """
    self.index = index
    self.net = None
    self.domain_size = None
    self.markov_blanket = None
    self.support = None
    self.name = name
    self._cpt_probabilities = None    
    self._distributions = None
    self._parent_multiplier = None    
    self._compiled = False
    if domain_size:
      self.set_domain_size(domain_size)
    if cpt_probabilities:
      self.set_cpt_probabilities(cpt_probabilities)
  
  def __str__(self):
    return "<{}>".format(self.name or self.index)

  def __repr__(self):
    return str(self)

  def __cmp__(self, other):
    return cmp(self.index, other.index)

  def __hash__(self):
    return self.index

  def _compute_markov_blanket(self):
    """Compute the Markov blanket for this node."""
    assert self.net
    coparents = [parent for child in self.children for parent in child.parents]
    overcomplete_blanket = list(self.parents) + list(self.children) + coparents
    markov_blanket = sorted(
      set(node for node in overcomplete_blanket if node != self),
      key=lambda node: node.index)
    return markov_blanket

  def _compute_distributions(self):
    """Compute the distribution for each setting of parent values."""
    assert self._cpt_probabilities
    assert self.net
    distributions = []
    parent_value_product = itertools.product(*[parent.support for parent in self.parents])
    j = 0
    for i, parent_values in enumerate(parent_value_product):
      values = self.support
      j = i * self.domain_size
      probabilities = self._cpt_probabilities[j:j+self.domain_size]
      distributions.append(
        distribution.CategoricalDistribution(values, probabilities, self.net.rng))
      world = dict(zip(self.parents, parent_values))
      assert(self._distribution_index(world) == i)
    assert j + self.domain_size == len(self._cpt_probabilities)
    return distributions

  def _compute_parent_multipliers(self):
    """Compute info used to associate parent values with distributions."""
    reversed_multiplier = []
    multiplier = 1
    for parent in reversed(self.parents):
      reversed_multiplier.append(multiplier)
      multiplier *= parent.domain_size
    parent_multiplier = list(reversed(reversed_multiplier))
    return parent_multiplier

  def _distribution_index(self, world):
    """Given parent values, return index that points to correct distribution."""
    return sum(world[parent]*self._parent_multiplier[i]
               for (i, parent) in enumerate(self.parents))
    
  def _get_distribution(self, world):
    """Given parent values in world, return appropriate distribution object."""
    return self._distributions[self._distribution_index(world)]

  def set_domain_size(self, domain_size):
    """Set the number of elements in the support of this node."""
    self.domain_size = domain_size

  def set_cpt_probabilities(self, cpt_probabilities):
    """Set the conditional probability table (CPT)."""
    self._cpt_probabilities = cpt_probabilities

  def set_net(self, net):
    """Set Bayes net associated with this node."""
    assert not self.net
    self.net = net    
    
  def compile(self):
    """Compute and store distributions and Markov blanket."""
    self.support = range(self.domain_size)    
    self.markov_blanket = self._compute_markov_blanket()
    self._parent_multiplier = self._compute_parent_multipliers()
    self._distributions = self._compute_distributions()
    self._compiled = True
    
  def sample(self, world):
    """Sample node value given parent values in world."""
    assert self._compiled
    return self._get_distribution(world).sample()

  def log_probability(self, world, node_value):
    """Return log probability of node value given parent values in world."""
    assert self._compiled
    return self._get_distribution(world).log_probability(node_value)

  @property
  def parents(self):
    return sorted(self.net.predecessors(self))

  @property
  def children(self):
    return sorted(self.net.successors(self))

  @property
  def cpt_probabilities(self):
    return self._cpt_probabilities


class BayesNet(networkx.DiGraph):
  """A Bayesian network.

  Creating a network:
  >>> from i3 import utils
  >>> from i3 import bayesnet
  >>> rng = utils.RandomState(seed=0)
  >>> node_1 = bayesnet.BayesNetNode(index=0, domain_size=2,
  ...   cpt_probabilities=[0.001, 0.999])
  >>> node_2 = bayesnet.BayesNetNode(index=1, domain_size=3, 
  ...   cpt_probabilities=[0.002, 0.008, 0.980, 0.980, 0.002, 0.008])  
  >>> net = bayesnet.BayesNet(rng,
  ...  nodes=[node_1, node_2],
  ...  edges=[(node_1, node_2)])
  >>> net.compile()

  Sampling random worlds:
  >>> world = net.sample()
  >>> world
  {<0>: 1, <1>: 0}

  Computing the (log) probabilities of random worlds:
  >>> net.log_probability(world)
  -0.011152871797601495
  """

  def __init__(self, rng, nodes=None, edges=None, **attr):
    """Initializes Bayesian network.

    Args:
      rng: a RandomState
      nodes: a list of BayesNetNodes
      edges: a list of pairs of BayesNetNodes
      attr: arguments parsed by networkx.DiGraph
    """
    super(BayesNet, self).__init__(**attr)
    self.rng = rng
    self.compiled = False
    self.nodes_by_index = []
    self.nodes_by_topology = None
    self.node_count = None
    if nodes:
      for node in nodes:
        self.add_node(node)
    if edges:
      self.add_edges_from(edges)    

  def __repr__(self):
    return str(self)

  def __str__(self):
    if not self.nodes_by_index:
      return "<<>>"
    s = "<<BN\n"
    for node in self.nodes_by_topology or self.nodes_by_index:
      s += "  {} -> {}  {} {}\n".format(
        node.parents, node, node.domain_size, node.cpt_probabilities)
    s += ">>"
    return s
      
  def compile(self):
    """Compute topological order, Markov blanket, etc."""
    self.nodes_by_topology = tuple(networkx.topological_sort(self))
    self.nodes_by_index = sorted(self.nodes(), key=lambda node: node.index)
    self.node_count = self.number_of_nodes()
    assert ([node.index for node in self.nodes_by_index] ==
            range(self.nodes_by_index[-1].index + 1))
    for node in self.nodes_by_topology:
      node.compile()
    self.compiled = True

  def sample(self, world=None):
    """Sample a random world, potentially based on existing world."""
    if world:
      world = world.copy()
    else:
      world = random_world.RandomWorld()
    for node in self.nodes_by_topology:
      if not node in world:
        world[node] = node.sample(world)
    return world

  def log_probability(self, world):
    """Return the log probability of this network returning the given world."""
    assert len(world) == self.node_count
    log_prob = 0.0
    for node in world:
      log_prob += node.log_probability(world, world[node])
    return log_prob

  def add_node(self, node, attr_dict=None, **attr):
    """Add node to network."""
    super(BayesNet, self).add_node(node, attr_dict=attr_dict, **attr)
    if self.nodes_by_index:
      assert node.index == self.nodes_by_index[-1].index + 1
    else:
      assert node.index == 0
    node.set_net(self)
    self.nodes_by_index.append(node)

  def find_node(self, name):
    """Return node with given name."""
    for node in self.nodes_by_index:
      if node.name == name:
        return node
    raise ValueError("Node with name {} not found!", name)


if __name__ == "__main__":
  import doctest
  doctest.testmod()
