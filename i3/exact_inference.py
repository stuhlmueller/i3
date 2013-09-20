"""Exact inference algorithms for Bayes nets."""
import math

from i3 import utils


class Enumerator(object):
  """Exact inference by enumeration."""

  def __init__(self, net):
    self.net = net

  def marginalize(self, evidence, query_node):
    """Compute marginal distribution on query variable given evidence.

    Args:
      net: a BayesNet
      evidence: a mapping from BayesNetNodes to values
      query_node: a BayesNetNode

    Returns:
      dictionary mapping values to probabilities (marginal dist)
    """
    assert query_node not in evidence
    log_probs = []
    values = query_node.support(evidence)
    for value in values:
      extended_evidence = evidence.extend(query_node, value)
      log_prob = self.marginalize_nodes(
        extended_evidence, self.net.sorted_nodes)
      log_probs.append(log_prob)
    probs = utils.normalize([math.exp(log_prob) for log_prob in log_probs])
    return dict(zip(values, probs))

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
      for value in node.support(evidence):
        local_logprob = node.log_probability(evidence, value)
        remainder_logprob = self.marginalize_nodes(
          evidence.extend(node, value),
          rest_nodes)
        log_probs.append(local_logprob + remainder_logprob)
      return utils.logsumexp(log_probs)
