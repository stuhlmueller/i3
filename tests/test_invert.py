import pytest

from i3 import invert
from i3.networks import sprinkler_net
from i3.networks import triangle_net
from i3 import utils


class TestSprinklerBayesNet(object):
  """Test inversion on a three-node rain/sprinkler/grass network."""

  def setup(self):
    self.rng = utils.RandomState(seed=0)
    self.net = sprinkler_net.get(self.rng)

  def test_detailed(self):
    """Test that inverses for sprinkler network have the correct structure."""
    evidence_nodes = [self.net.find_node("Grass")]
    rain_node = self.net.find_node("Rain")
    sprinkler_node = self.net.find_node("Sprinkler")
    grass_node = self.net.find_node("Grass")
    inverse_map = invert.compute_inverse_map(
      self.net, evidence_nodes, self.rng)
    assert len(inverse_map) == 2
    for (final_node, inverse_net) in inverse_map.items():
      inv_rain_node = inverse_net.find_node("Rain")
      inv_sprinkler_node = inverse_net.find_node("Sprinkler")
      inv_grass_node = inverse_net.find_node("Grass")
      inv_final_node = inverse_net.nodes_by_index[final_node.index]
      assert len(inv_grass_node.parents) == 0
      assert len(inv_final_node.parents) == 2
      assert inv_rain_node.support == rain_node.support
      assert inv_sprinkler_node.support == sprinkler_node.support
      assert inv_grass_node.support == grass_node.support
      for node in inverse_net.nodes_by_index:
        if node != inv_grass_node and node != inv_final_node:
          assert node.parents == [inv_grass_node]

  @pytest.mark.parametrize("evidence_index", [0, 1, 2, 3])
  def test_broad(self, evidence_index):
    """Test inversion across different evidence settings."""
    evidence = sprinkler_net.evidence(evidence_index)
    evidence_nodes = [self.net.nodes_by_index[index]
                      for index in evidence.keys()]
    inverse_map = invert.compute_inverse_map(
      self.net, evidence_nodes, self.rng)
    assert len(inverse_map) == 2
    for (final_node, inverse_net) in inverse_map.items():
      assert final_node.index not in evidence
      inv_final_node = inverse_net.nodes_by_index[final_node.index]
      assert len(inv_final_node.parents) == 2
      for evidence_node in evidence_nodes:
        inv_evidence_node = inverse_net.nodes_by_index[evidence_node.index]
        assert len(inv_evidence_node.parents) == 0


class TestTriangleNet(object):
  """Test inversion on 120-node UAI network."""

  def setup(self):
    self.rng = utils.RandomState(seed=0)
    self.net = triangle_net.get(self.rng)

  @pytest.mark.parametrize("max_inverse_size", [1, 2, 3, 5, 10, 20, 120])
  def test_max_inverse_size(self, max_inverse_size):
    evidence = triangle_net.evidence(0)
    evidence_nodes = [self.net.nodes_by_index[index]
                      for index in evidence.keys()]
    inverse_map = invert.compute_inverse_map(
      self.net, evidence_nodes, self.rng, max_inverse_size=max_inverse_size)
    assert len(inverse_map) == self.net.node_count - len(evidence_nodes)
    for fwd_final_node, inverse_net in inverse_map.items():
      inverse_net.compile()
      inv_final_node = inverse_net.nodes_by_index[fwd_final_node.index]
      assert inverse_net.nodes_by_topology[-1] == inv_final_node
      assert inv_final_node.parents
      num_nodes_with_parents = 0
      for node in inverse_net.nodes_by_index:
        if node.parents:
          num_nodes_with_parents += 1
      assert (num_nodes_with_parents ==
              min(max_inverse_size, self.net.node_count - len(evidence_nodes)))


def test_distance_scorer():
  rng = utils.RandomState(seed=0)
  net = triangle_net.get(rng)
  nodes = net.nodes_by_index
  for i in [1, 2]:
    start_nodes = nodes[:i]
    scorer = invert.distance_scorer(net, start_nodes)
    end_nodes = [node for node in nodes if node not in start_nodes]
    for end_node in end_nodes:
      for node in nodes:
        score = scorer(node, end_node)
        if node == end_node:
          assert score == float("-inf")
        elif node in start_nodes:
          assert score == float("+inf")
        else:
          assert float("-inf") < score < float("+inf")
