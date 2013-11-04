"""Test import of UAI Bayes net files."""

import numpy as np

from i3 import random_world
from i3 import uai_import
from i3 import utils
from i3.networks import triangle_net


NETWORK_STRING_A = """BAYES
3
2 2 3
3
1 0
2 0 1
2 1 2

2
 0.436 0.564

4
 0.128 0.872
 0.920 0.080

6
 0.210 0.333 0.457
 0.811 0.000 0.189"""

NETWORK_STRING_B = """BAYES
3
2 2 3
3
1 0
2 1 2
2 0 1

2
 0.436 0.564

4
 0.128 0.872
 0.920 0.080

6
 0.210 0.333 0.457
 0.811 0.000 0.189"""

NETWORK_PROBABILITIES = [
  ([0, 0, 0], 0.436 * 0.128 * 0.210),
  ([1, 1, 2], 0.564 * 0.080 * 0.189),
  ([1, 0, 2], 0.564 * 0.920 * 0.457),
  ([0, 0, 1], 0.436 * 0.128 * 0.333),
  ([0, 1, 1], 0.436 * 0.872 * 0.000),
]

EVIDENCE_STRING = """2 1 0 2 1"""

MARGINAL_STRING = """MAR
1
3 2 0.5 0.5 2 0.354608 0.645392 2 0.254609 0.745391
-BEGIN-
1
3 2 0.084297 0.915703 2 0.354608 0.645392 2 0.254609 0.745391"""

MARGINAL_PROBABILITIES = {
  0: [0.084297, 0.915703],
  1: [0.354608, 0.645392],
  2: [0.254609, 0.745391]
}


class TestCPTReordering(object):
  def test_v1(self):
    probs = [0.210, 0.333, 0.457, 0.811, 0.000, 0.189]
    old_order = [0, 1]
    new_order = [1, 0]
    domain_sizes = [2, 3]
    assert (uai_import.reorder_cpt(old_order, domain_sizes, probs, new_order) ==
            [0.210, 0.811, 0.333, 0.000, 0.457, 0.189])

  def test_v2(self):
    probs = [.1, .2, .3, .4, .5, .6, .7, .8]
    old_order = [0, 1, 2]
    new_order = [2, 0, 1]
    domain_sizes = [2, 2, 2]
    assert (uai_import.reorder_cpt(old_order, domain_sizes, probs, new_order) ==
            [.1, .3, .5, .7, .2, .4, .6, .8])


class TestNetworkImport(object):
  def test_string_import(self):
    """Check that probabilities for imported Bayes net are as expected."""
    for network_string in [NETWORK_STRING_A, NETWORK_STRING_B]:
      net = uai_import.network_from_string(network_string, utils.RandomState())
      assert len(net.sample()) == 3
      for (values, prob) in NETWORK_PROBABILITIES:
        world = random_world.RandomWorld(
          keys=net.nodes_by_index,
          values=values)
        np.testing.assert_almost_equal(
          net.log_probability(world),
          utils.safe_log(prob))

  def test_file_import(self):
    """Check that importing big files doesn't throw errors."""
    rng = utils.RandomState(seed=0)
    triangle_net.get(rng)
    triangle_net.evidence(0)
    triangle_net.marginals(0)


class TestEvidenceImport(object):
  def test_string_import(self):
    """Check that imported random worlds look as expected."""
    world = uai_import.evidence_from_string(EVIDENCE_STRING)
    assert world[1] == 0
    assert world[2] == 1


class TestMarginalImport(object):
  def test_string_import(self):
    """Check that imported marginals look as expected."""
    marginals = uai_import.marginals_from_string(MARGINAL_STRING)
    for (index, probs) in marginals.items():
      np.testing.assert_array_almost_equal(
        probs, MARGINAL_PROBABILITIES[index])


class TestToken(object):
  def test_token(self):
    """Check that funcparserlib tokens work as expected."""
    tokens = uai_import.string_to_tokens("foo bar baz")
    assert (repr(tokens[0]) ==
            "Token('NAME', 1, 'foo', (1, 0), (1, 3), 'foo bar baz')")
    assert len(tokens) == 4
    assert tokens[-1].type == 'ENDMARKER'
