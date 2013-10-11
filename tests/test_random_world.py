"""Test random world data structure."""
from i3 import random_world
from i3 import bayesnet


def test_random_world():
  a, b, c = [bayesnet.BayesNetNode(i) for i in [0, 1, 2]]
  # Empty world
  world = random_world.RandomWorld()
  assert not world
  assert len(world) == 0
  assert len(world.copy()) == 0
  assert len(world.extend(a, 1)) == 1
  world[b] = 2
  assert len(world) == 1
  assert world[b] == 2
  assert world.items() == [(1, 2)]
  assert b in world
  assert a not in world
  assert world
  assert list(world) == [1]
  del world[b]
  assert b not in world
  # Nonempty world
  nodes = [a, b, c]
  values = [True, False, 3]
  world = random_world.RandomWorld(nodes, values)
  assert set(world.keys()) == {0, 1, 2}
  assert set(world.values()) == {True, False, 3}
  assert world
  assert len(world) == 3
  for node, value in zip(nodes, values):
    assert node in world
    assert world[node] == value
  assert sorted(list(world)) == [0, 1, 2]
  assert str(world) == repr(world)
