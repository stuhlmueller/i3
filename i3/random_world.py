"""Random worlds map BayesNetNodes to values."""
import itertools
import pprint
import numpy as np


class RandomWorld(object):
  """A mapping from BayesNetNodes to values."""

  def __nonzero__(self):
    return not np.all(self.data == -1)

  def __contains__(self, node):
    return self.data[node.index] != -1

  def __delitem__(self, node):
    self.data[node.index] = -1

  def __getitem__(self, node):
    return self.data[node.index]

  def __init__(self, obj):
    """
    Args:
      x: either array of values or sizes of world
    """
    if hasattr(obj, "__iter__"):
      self.data = np.array(obj, dtype=np.int8)
    else:
      self.data = np.ones(obj, dtype=np.int8) * -1

  def __iter__(self):
    return self.data.__iter__()

  def __len__(self):
    return len(self.data)    

  def __setitem__(self, node, value):
    self.data[node.index] = value

  def __str__(self):
    return "{{W {}}}".format(list(self.data))

  def __repr__(self):
    return str(self)

  def copy(self):
    """Return a copy of this random world."""
    return RandomWorld(self.data.copy())

  def extend(self, node, value):
    """Return an extended copy of this random world."""
    assert node not in self
    copy_world = self.copy()
    copy_world[node] = value
    return copy_world

  def set_index_value(self, index, value):
    """Directly set world value using index."""
    self.data[index] = value

  def get_index_value(self, index):
    """Directly get world value using index."""
    return self.data[index]


def all_random_worlds(nodes, num_nodes_total):
  """Return iterable over all possible random worlds.

  Args:
    nodes: a list of nodes
    num_nodes_total: total number of nodes in network

  Returns:
    iterable of random worlds mapping nodes to values
  """
  for values in itertools.product(*[node.support for node in nodes]):
    world = RandomWorld(num_nodes_total)
    for (node, value) in zip(nodes, values):
      world[node] = value
    yield world
    
