from __future__ import division

import collections
import networkx as nx
import numpy as np

from i3 import bayesnet


def dependent_nodes(node, observed_nodes):
  """Recursively compute all nodes that probabilistically depend on node."""
  dependencies = set()
  queue = collections.deque([node])
  while queue:
    next_node = queue.popleft()
    if next_node not in dependencies:
      dependencies.add(next_node)
      assert next_node.markov_blanket is not None
      queue.extend(next_node.markov_blanket)
  return dependencies & observed_nodes


def distance_scorer(net, start_nodes):
  """Build a distance-based scorer for Bayes net nodes.

  Build a scorer for (start_nodes, node, end_node) triples such that
  the score is highest if node is close to start_nodes and lowest
  if node is close to end_node.

  Args:
    net: a BayesNet
    start_nodes: a list of BayesNetNodes that are part of net

  Returns:
    scorer: a function that maps pairs of (node, end_node) to real numbers.
  """
  undirected_net = nx.Graph(net)

  # All-pairs shortest-path
  path_length = nx.shortest_path_length(undirected_net)

  # For each latent node, compute average of min distances to start nodes.
  start_distance = {}
  for node in net.nodes_by_index:
    distances = [path_length[start_node][node] for start_node in start_nodes]
    start_distance[node] = np.mean(distances)

  max_start_distance = max(start_distance.values())

  def scorer(node, end_node):
    """Return score for node given (fixed) start_nodes and end_nodes"""
    if node == end_node:
      score = float("-inf")
    elif node in start_nodes:
      score = float("inf")
    else:
      score = ((max_start_distance - start_distance[node]) +
               path_length[node][end_node])
    return score

  return scorer


def compute_inverse_net(net, start_nodes, end_node, rng, max_inverse_size):
  """Compute a single Bayes net inverse given start and end nodes.

  Args:
    net: a BayesNet
    start_nodes: a list of BayesNetNodes in net
    end_node: a BayesNetNode
    rng: a RandomState
    max_inverse_size: going topologically backwards from end node,
      only include up to this number of nodes in computed inverse
      edges
            
  Returns:
    inverse_net: a BayesNet with the same number of nodes as input net
  """  
  for node in start_nodes + [end_node]:
    assert node in net.nodes_by_index
    assert node.support

  # Compute distance-based scorer for Bayes net nodes.
  scorer = distance_scorer(net, start_nodes)

  # Sort nodes s.t. nodes close to evidence are first, focal node is last.
  scored_indices = []
  for node in net.nodes_by_index:
    scored_indices.append((node.index, scorer(node, end_node)))
  sorted_pairs = sorted(scored_indices, key=lambda (i, s): -s)
  sorted_indices = [index for (index, _) in sorted_pairs]

  # Create inverse net.
  inverse_nodes = [
    bayesnet.DistBayesNetNode(
      node.index, name=node.name, domain_size=node.domain_size)
    for node in net.nodes_by_index]
  inverse_net = bayesnet.BayesNet(rng, nodes=inverse_nodes)

  # Add edges (based on dependencies in fwd net).
  observed_nodes = set(start_nodes)
  for index in sorted_indices:
    forward_node = net.nodes_by_index[index]
    if forward_node not in observed_nodes:
      if net.node_count - index <= max_inverse_size:
        forward_deps = dependent_nodes(forward_node, observed_nodes)
        for forward_dep in forward_deps:
          inverse_net.add_edge(
            inverse_nodes[forward_dep.index], inverse_nodes[index])
      observed_nodes.add(forward_node)

  # FIXME: the inverse networks should only contain relevant nodes.
  
  return inverse_net


def compute_inverse_map(net, evidence_nodes, rng, max_inverse_size=None):
  """Given evidence nodes, compute one inverse for each latent node in net.

  Args:
    net: a BayesNet
    evidence_nodes: a list of BayesNetNodes in net
    rng: a RandomState
    max_inverse_size: going topologically backwards from end node,
      only include up to this number of nodes in computed inverse
      edges
  
  Returns:
    inverse_map: a BayesNetMap
  """
  if max_inverse_size is None:
    max_inverse_size = net.node_count  
  inverse_map = bayesnet.BayesNetMap()
  for node in net.nodes_by_index:
    if node not in evidence_nodes:
      inverse_net = compute_inverse_net(
        net, evidence_nodes, node, rng, max_inverse_size)
      inverse_map.add_net(node, inverse_net)
  return inverse_map
