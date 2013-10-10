from i3 import evid
from i3 import invert
from i3.networks import sprinkler_net
from i3 import utils


def test_invert_sprinkler():
  """Test that inverses for sprinkler network have the correct structure."""
  rng = utils.RandomState(seed=0)
  net = sprinkler_net.get(rng)
  evidence_nodes = [net.find_node("Grass")]
  rain_node = net.find_node("Rain")
  sprinkler_node = net.find_node("Sprinkler")
  grass_node = net.find_node("Grass")
  inverse_map = invert.compute_inverse_map(net, evidence_nodes, rng)
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
