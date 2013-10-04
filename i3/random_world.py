"""Random worlds map BayesNetNodes to values."""
import itertools
import pprint


def as_index(node_or_index):
  if hasattr(node_or_index, "index"):
    return node_or_index.index
  else:
    return node_or_index


class RandomWorld(object):
  """A mapping from BayesNetNodes to values."""

  def __nonzero__(self):
    return bool(self.data)

  def __contains__(self, key):
    return as_index(key) in self.data

  def __delitem__(self, key):
    del self.data[as_index(key)]

  def __getitem__(self, key, default=None):
    return self.data.get(as_index(key), default)

  def __init__(self, keys=None, values=None):
    if keys or values:
      assert len(keys) == len(values)
      self.data = dict(zip([as_index(key) for key in keys], values))
    else:
      self.data = {}

  def __iter__(self):
    return self.data.__iter__()

  def __len__(self):
    return len(self.data)    

  def __setitem__(self, key, value):
    self.data[as_index(key)] = value

  def __str__(self):
    return pprint.pformat(self.data)

  def __repr__(self):
    return str(self)

  def copy(self):
    """Return a copy of this random world."""
    items = self.data.items()
    keys = [key for (key, value) in items]
    values = [value for (key, value) in items]
    return RandomWorld(keys, values)

  def extend(self, key, value):
    """Return an extended copy of this random world."""
    assert as_index(key) not in self.data
    copy_world = self.copy()
    copy_world[as_index(key)] = value
    return copy_world
  
  def items(self):
    """Return a list of key-value pairs."""
    return self.data.items()

  def keys(self):
    """Return a copy of random world's keys (indices)."""
    return self.data.keys()

  def values(self):
    """Return a copy of random world's values."""    
    return self.data.values()


def all_random_worlds(variables):
  """Return iterable over all possible random worlds.

  Args:
    variables: a list of variables

  Returns:
    iterable of random worlds mapping variables to values
  """
  for values in itertools.product(*[var.support for var in variables]):
    yield RandomWorld(variables, values)
    
