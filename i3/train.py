from __future__ import division

from i3 import gibbs
from i3 import learn


class Trainer(object):
  """Learn distributions for all conditionals in a BayesNetMap."""

  def __init__(self, net, inverse_map, precompute_gibbs):
    """Extracting all distinct conditionals from inverse map.

    Args:
      net: a BayesNet
      inverse_map: a BayesNetMap with inverses for the Bayes net
      precompute_gibbs: a Boolean indicating whether to do exact
        computation of Gibbs conditinoals during initialization.
    """
    self.net = net
    self.learners = {}
    self.inverse_map = inverse_map
    for inverse_net in inverse_map.values():
      for node in inverse_net.nodes_by_index:
        parents = node.parents
        parent_indices = tuple([parent.index for parent in parents])
        key = (parent_indices, node.index)
        if node.children or not precompute_gibbs:
          learner = self.learners.setdefault(
            key, learn.CountLearner(node.support, net.rng))
        else:
          forward_node = net.nodes_by_index[node.index]
          learner = self.learners.setdefault(
            key, learn.GibbsLearner(forward_node, net.rng))
        node.set_distribution(learner)

  def observe(self, world):
    """Update learners given random world."""
    for ((parent_indices, node_index), learner) in self.learners.items():
      parent_values = [world.data[index] for index in parent_indices]
      node_value = world.data[node_index]
      learner.observe(parent_values, node_value)
  
  def finalize(self):
    """Finalize learners, compile network."""
    for learner in self.learners.values():
      learner.finalize()
    for net in self.inverse_map.values():
      net.compile()
