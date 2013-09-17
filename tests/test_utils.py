#!/usr/bin/env python

"""
Tests for sampling and probability calculation utilities.
"""

import math
import numpy as np

from i3 import utils

def test_safelog_base():
  """Test that safe_log does not differ from log for values > 0."""
  nums = [1, 10, 100, 1000, 10000]
  np.testing.assert_array_almost_equal(
    [utils.safe_log(num) for num in nums],
    [math.log(num) for num in nums])

def test_safelog_extended():
  """Test that safe_log returns -inf for 0."""
  assert utils.safe_log(0) == utils.NEGATIVE_INFINITY
  assert utils.NEGATIVE_INFINITY == float("-inf")
