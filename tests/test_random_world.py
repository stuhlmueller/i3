"""Test random world data structure."""
from i3 import random_world
from i3 import bayesnet


def test_random_world():
  A, B, C = [bayesnet.BayesNetNode(i) for i in [0, 1, 2]]
  # Empty world
  world = random_world.RandomWorld()
  assert not world
  assert len(world) == 0
  assert len(world.copy()) == 0
  assert len(world.extend(A, 1)) == 1
  world[B] = 2
  assert len(world) == 1
  assert world[B] == 2
  assert world.items() == [(1, 2)]
  assert B in world
  assert A not in world
  assert world
  assert list(world) == [1]
  del world[B]
  assert B not in world
  # Nonempty world
  nodes = [A, B, C]
  values = [True, False, 3]
  world = random_world.RandomWorld(nodes, values)
  assert world
  assert len(world) == 3
  for node, value in zip(nodes, values):
    assert node in world
    assert world[node] == value
  assert sorted(list(world)) == [0, 1, 2]
  assert str(world) == repr(world)
