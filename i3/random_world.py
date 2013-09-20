"""Random worlds map BayesNetNodes to values."""
import itertools


class RandomWorld(object):
  """A mapping from BayesNetNodes to values."""

  def __bool__(self):
    return bool(self.data)

  def __contains__(self, key):
    return key in self.data

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


def all_random_worlds(variable_names, support):
  """Return iterable over all possible random worlds.

  Args:
    variable_names: a list of variable names
    support: a list of possible values

  Returns:
    iterable of dictionaries mapping variables to values
  """
  for values in itertools.product(*[support]*len(variable_names)):
    yield RandomWorld(variable_names, values)
    
