from __future__ import division

import numpy as np
import pytest

from i3 import marg
from i3 import random_world
from i3 import utils
from i3.networks import sprinkler_net


def test_marg():
  marginals_a = marg.Marginals(
    [0, 1, 2],
    [[.1, .9],
     [.2, .8],
     [.3, .7]])
  marginals_b = marg.Marginals(
    [0, 1, 2],
    [[.11, .89],
     [.19, .81],
     [.31, .69]])
  assert marginals_a - marginals_b < .02
  assert marginals_a - marginals_b <= .02
  assert marginals_a - marginals_b > .001
  assert marginals_a - marginals_b >= .001
  assert not marginals_a - marginals_b < .01
  assert marginals_a > None
  assert not marginals_a < None


def test_marg_counter():
  rng = utils.RandomState(seed=0)
  net = sprinkler_net.get(rng)
  counter = marg.MarginalCounter(net)
  with pytest.raises(AssertionError):
    counter.marginals()
  value_lists = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 1]]
  for values in value_lists:
    world = random_world.RandomWorld(
      [0, 1, 2],
      values)
    counter.observe(world)
  assert counter.num_observations == 3
  marginals = counter.marginals()
  np.testing.assert_almost_equal(marginals[0][0], 1.0)
  np.testing.assert_almost_equal(marginals[0][1], 0.0)
  np.testing.assert_almost_equal(marginals[1][0], 2/3)
  np.testing.assert_almost_equal(marginals[1][1], 1/3)
  np.testing.assert_almost_equal(marginals[2][0], 1/3)
  np.testing.assert_almost_equal(marginals[2][1], 2/3)    
  
  
