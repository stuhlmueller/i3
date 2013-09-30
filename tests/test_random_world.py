"""Test random world data structure."""
from i3 import random_world

def test_random_world():
  # Empty world
  world = random_world.RandomWorld()
  assert not world
  assert len(world) == 0
  assert len(world.copy()) == 0
  assert len(world.extend("A", 1)) == 1
  world["B"] = 2
  assert len(world) == 1
  assert world["B"] == 2
  assert world.items() == [("B", 2)]
  assert "B" in world
  assert "A" not in world
  assert world
  assert list(world) == ["B"]
  del world["B"]
  assert "B" not in world
  # Nonempty world
  nodes = ["A", "B", "C"]
  values = [True, False, 3]
  world = random_world.RandomWorld(nodes, values)
  assert world
  assert len(world) == 3
  for node, value in zip(nodes, values):
    assert node in world
    assert world[node] == value
  assert sorted(list(world)) == ["A", "B", "C"]
  assert str(world) == repr(world)
