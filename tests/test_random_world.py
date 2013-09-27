"""Test random world data structure."""
import numpy as np

from i3 import bayesnet
from i3 import random_world


class TestRandomWorld():

  def test_a(self):
    world = random_world.RandomWorld(3)
    assert not world
    assert len(world) == 3
    assert len(world.copy()) == 3
    A = bayesnet.BayesNetNode(0)
    B = bayesnet.BayesNetNode(1)
    assert len(world.extend(A, 1)) == 3
    world[B] = 2
    assert len(world) == 3
    assert world[B] == 2
    assert B in world
    assert A not in world
    assert world
    assert list(world) == [-1, 2, -1]

  def test_b(self): 
    nodes = [bayesnet.BayesNetNode(i) for i in [0, 1, 2]]
    values = [True, False, 3]
    world = random_world.RandomWorld(values)
    assert world
    assert len(world) == 3
    for node, value in zip(nodes, values):
      assert node in world
      assert world[node] == value
    assert sorted(list(world)) == [False, True, 3]
    assert str(world) == repr(world)
