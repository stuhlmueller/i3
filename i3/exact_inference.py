"""Exact inference algorithms for Bayes nets."""
import math

from i3 import marg
from i3 import utils


class Enumerator(object):
  """Exact inference by enumeration."""

  def __init__(self, net, evidence):
    self.net = net
    self.evidence = evidence

  def marginals(self):
    """Compute marginal distribution for all non-evidence nodes."""
    # FIXME: Inefficient
    dists = marg.Marginals()
    for index, value in self.evidence.items():
      dists[index] = [0] * self.net.nodes_by_index[index].domain_size
      dists[index][value] = 1
    for node in self.net.nodes_by_index:
      if node not in self.evidence:
        dists[node] = self.marginalize_node(node)
    return dists

  def marginalize_node(self, query_node):
    """Compute marginal distribution on query variable given evidence.

    Args:
      net: a BayesNet
      evidence: a mapping from BayesNetNodes to values
      query_node: a BayesNetNode

    Returns:
      list of probabilities (marginal dist)
    """
    assert query_node not in self.evidence
    log_probs = []
    values = query_node.support
    for value in values:
      extended_evidence = self.evidence.extend(query_node, value)
      log_prob = self.marginalize_nodes(
        extended_evidence, self.net.nodes_by_topology)
      log_probs.append(log_prob)
    probs = utils.normalize([math.exp(log_prob) for log_prob in log_probs])
    return probs

  def marginalize_nodes(self, evidence, nodes):
    """Compute the normalization constant of nodes given evidence.

    Args:
      evidence: mapping from BayesNetNodes to values
      nodes: list of BayesNetNodes

    Returns:
      a log-probability
    """
    if not nodes:
      return utils.LOG_PROB_1
    node, rest_nodes = nodes[0], nodes[1:]
    if node in evidence:
      local_logprob = node.log_probability(evidence, evidence[node])
      remainder_logprob = self.marginalize_nodes(evidence, rest_nodes)
      return local_logprob + remainder_logprob
    else:
      log_probs = []
      for value in node.support:
        local_logprob = node.log_probability(evidence, value)
        remainder_logprob = self.marginalize_nodes(
          evidence.extend(node, value),
          rest_nodes)
        log_probs.append(local_logprob + remainder_logprob)
      return utils.logsumexp(log_probs)
