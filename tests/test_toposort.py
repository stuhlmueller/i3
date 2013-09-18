"""Tests for topological sorting."""

from i3 import toposort


def test_toposort():
  """Test that sorted order is actually topological."""
  edges = [(0, 1), (1, 2), (0, 2), (4, 3), (3, 0)]
  ordered_nodes = toposort.toposort(edges)
  assert ordered_nodes == [4, 3, 0, 1, 2]
