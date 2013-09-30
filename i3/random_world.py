"""Random worlds map BayesNetNodes to values."""
import itertools
import pprint


class RandomWorld(object):
  """A mapping from BayesNetNodes to values."""

  def __nonzero__(self):
    return bool(self.data)

  def __contains__(self, key):
    return key in self.data

  def __delitem__(self, key):
    del self.data[key]

  def __getitem__(self, key):
    return self.data[key]    

  def __init__(self, nodes=None, values=None):
    if nodes or values:
      assert len(nodes) == len(values)
      self.data = dict(zip(nodes, values))
    else:
      self.data = {}

  def __iter__(self):
    return self.data.__iter__()

  def __len__(self):
    return len(self.data)    

  def __setitem__(self, key, value):
    self.data[key] = value

  def __str__(self):
    return pprint.pformat(self.data)

  def __repr__(self):
    return str(self)

  def copy(self):
    """Return a copy of this random world."""
    items = self.data.items()
    nodes = [node for (node, value) in items]
    values = [value for (node, value) in items]
    return RandomWorld(nodes, values)

  def extend(self, node, value):
    """Return an extended copy of this random world."""
    assert node not in self.data
    copy_world = self.copy()
    copy_world[node] = value
    return copy_world

  def items(self):
    """Return a list of key-value pairs."""
    return self.data.items()


def all_random_worlds(variables):
  """Return iterable over all possible random worlds.

  Args:
    variables: a list of variables

  Returns:
    iterable of random worlds mapping variables to values
  """
  for values in itertools.product(*[var.support for var in variables]):
    yield RandomWorld(variables, values)
    
