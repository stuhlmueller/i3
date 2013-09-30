import StringIO
import collections
import operator
import pprint
import token
import tokenize
from funcparserlib import parser

from i3 import bayesnet
from i3 import random_world
from i3 import utils


class Token(object):
  """A multi-character element in a string, with position and type."""
  
  def __init__(self, code, value, start=(0, 0), stop=(0, 0), line=''):
    self.code = code
    self.value = value
    self.start = start
    self.stop = stop
    self.line = line

  @property
  def type(self):
    return token.tok_name[self.code]

  def __unicode__(self):
    pos = '-'.join('%d,%d' % x for x in [self.start, self.stop])
    return "%s %s '%s'" % (pos, self.type, self.value)

  def __repr__(self):
    return 'Token(%r, %i, %r, %r, %r, %r)' % (
      self.type, self.code, self.value, self.start, self.stop, self.line)

  def __eq__(self, other):
    return (self.code, self.value) == (other.code, other.value)


def string_to_tokens(s):
  """A lexer that parses a string into a list of tokens."""
  return list(
    Token(*t)
    for t in tokenize.generate_tokens(StringIO.StringIO(s).readline)
    if t[0] not in [token.NEWLINE])


def string_to_number(s):
  """Convert string to integer or float."""
  try:
    return int(s)
  except ValueError:
    return float(s)


def token_value(t):
  """Return the value attribute of a token."""
  return t.value


def remove_ignored(tokens):
  """Remove all _Ignored tokens."""
  return [t for t in tokens if not isinstance(t, parser._Ignored)]


def number_parser():
  """Return parser that reads (float and int) numbers with whitespace."""
  number = (parser.some(lambda tok: tok.type == 'NUMBER')
            >> token_value
            >> string_to_number)
  indent = parser.some(lambda t: t.code == token.INDENT)
  dedent = parser.a(Token(token.DEDENT, ''))
  newline = parser.a(Token(54, '\n'))
  ignored_whitespace = parser.skip(indent | dedent | newline)
  return parser.oneplus(number | ignored_whitespace)


def end_parser():
  endmark = parser.a(Token(token.ENDMARKER, ''))
  end = parser.skip(endmark + parser.finished)
  return end


def parse_network(s):
  """Turn a net string into numbers describing a Bayes net."""
  net_type = parser.skip(parser.a(Token(token.NAME, 'BAYES')))
  network_parser = net_type + number_parser() + end_parser()
  tokens = string_to_tokens(s)
  return collections.deque(remove_ignored(network_parser.parse(tokens)))


def parse_evidence(s):
  """Turn an evidence string into numbers describing a random world."""
  evidence_parser = number_parser() + end_parser()
  tokens = string_to_tokens(s)
  return collections.deque(remove_ignored(evidence_parser.parse(tokens)))


def reorder_cpt(old_order, old_domain_sizes, old_probs, new_order):
  """
  Args:
    old_order: list of indices
    old_domain_sizes: mapping from indices to integers
    old_probs: list of probabilities  
    new_order: list of indices

  Example:
    old_order = [a, b, c]
    old_domain_sizes = [2, 2, 2]  
    old_probs = [.1, .2, .3, .4, .5, .6, .7, .8]
    new_order = [c, a, b]

  Old (unnormalized) CPT:
    a=0, b=0, c=0 .1
    a=0, b=0, c=1 .2
    a=0, b=1, c=0 .3
    a=0, b=1, c=1 .4
    a=1, b=0, c=0 .5
    a=1, b=0, c=1 .6
    a=1, b=1, c=0 .7
    a=1, b=1, c=1 .8  

  New (unnormalized) CPT:
    c=0, a=0, b=0: .1
    c=0, a=0, b=1: .3
    c=0, a=1, b=0: .5
    c=0, a=1, b=1: .7
    c=1, a=0, b=0: .2
    c=1, a=0, b=1: .4
    c=1, a=1, b=0: .6
    c=1, a=1, b=1: .8
  """  
  assert len(old_order) == len(old_domain_sizes)
  assert set(old_order) == set(new_order)
  assert len(old_probs) == reduce(operator.mul, old_domain_sizes, 1)
  
  # Create a mapping from old-order value tuples to probabilities.
  old_domains = [range(domain_size) for domain_size in old_domain_sizes]
  old_value_lists = utils.lexicographic_combinations(old_domains)
  old_value_tuples = [tuple(values) for values in old_value_lists]
  assert len(old_value_tuples) == len(old_probs)
  old_value_tuple_to_prob = dict(zip(old_value_tuples, old_probs))

  # Create mapping from old to new clique member indices.
  old_to_new_index = dict(zip(old_order, new_order))

  # Create mapping from old clique index to domain
  old_index_to_domain = dict(zip(old_order, old_domains))

  # Iterate through new-order value tuples, create reordered list of
  # probabilities.
  new_domains = [old_index_to_domain[old_to_new_index[old_index]]
                 for old_index in old_order]
  new_probs = []
  for new_value_list in utils.lexicographic_combinations(new_domains):
    new_value_tuple = tuple(new_value_list)
    old_value_tuple = tuple(
      utils.reordered_list(new_order, old_order, new_value_tuple))
    prob = old_value_tuple_to_prob[old_value_tuple]
    new_probs.append(prob)

  return new_probs
  

def evaluate_network(rng, stack):
  """Turn random state and stack of (UAI) net numbers into Bayesian network."""
  
  net = bayesnet.BayesNet(rng=rng)
  
  num_vars = stack.popleft()
  
  for i in range(num_vars):
    domain_size = stack.popleft()
    print "Adding node {} ({} elements in support)".format(i, domain_size)
    node = bayesnet.BayesNetNode(index=i)
    node.set_domain_size(domain_size)
    net.add_node(node)
    
  num_cliques = stack.popleft()
  assert num_cliques == num_vars

  clique_order = {}
  for node in net.nodes_by_index:
    clique_size = stack.popleft()
    clique_order[node] = utils.pop_n(stack, clique_size)
    for index in clique_order[node]:
      if index != node.index:
        print "Adding edge from node {} to node {}".format(index, node.index)
        net.add_edge(net.nodes_by_index[index], node)
  
  for node in net.nodes_by_index:
    cpt_size = stack.popleft()
    cpt_probabilities = utils.pop_n(stack, cpt_size)
    for cpt_prob in cpt_probabilities:
      assert 0 <= cpt_prob <= 1
    old_domain_sizes = [net.nodes_by_index[i].domain_size
                        for i in clique_order[node]]
    parent_indices = sorted([i for i in clique_order[node] if i != node.index])
    new_clique_order = parent_indices + [node.index]
    new_cpt_probabilities = reorder_cpt(
      clique_order[node], old_domain_sizes, cpt_probabilities, new_clique_order)
    node.set_cpt_probabilities(new_cpt_probabilities)
  
  assert not stack
  net.compile()
  return net


def evaluate_evidence(stack):
  """Turn stack of (UAI) evidence numbers into list of random worlds."""
  samples = []
  num_samples = stack.popleft()
  for i in range(num_samples):
    num_variables = stack.popleft()
    world = random_world.RandomWorld()
    for j in range(num_variables):
      index = stack.popleft()
      value = stack.popleft()
      world[index] = value
    samples.append(world)
  assert not stack
  return samples


def network_from_string(s, rng):
  """Parse (UAI) network string to network."""
  stack = parse_network(s)
  net = evaluate_network(rng, stack)
  return net  


def evidence_from_string(s):
  """Parse (UAI) evidence string to evidence (list of random worlds)."""
  stack = parse_evidence(s)
  random_worlds = evaluate_evidence(stack)
  return random_worlds


def load_network(filename, rng):
  """Parse (UAI) file to network."""  
  s = open(filename).read()
  return network_from_string(s, rng)
