"""Tests for topological sorting."""

import itertools

from i3 import toposort
from i3 import utils


def test_toposort():
  """Test that sorted order is actually topological."""
  edges = [(0, 1), (1, 2), (0, 2), (4, 3), (3, 0)]
  for shuffled_edges in itertools.permutations(edges):
    ordered_nodes = toposort.toposort(shuffled_edges)
    assert ordered_nodes == [4, 3, 0, 1, 2]


def test_sprinkler():
  """Test toposort on sprinkler network."""
  edges = [("rain", "sprinkler"), ("rain", "grass"), ("sprinkler", "grass")]
  for shuffled_edges in itertools.permutations(edges):
    ordered_nodes = toposort.toposort(shuffled_edges)
    assert ordered_nodes == ["rain", "sprinkler", "grass"]
