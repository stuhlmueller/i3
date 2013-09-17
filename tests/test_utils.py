#!/usr/bin/env python

"""
Tests for sampling and probability calculation utilities.
"""

import math
import unittest

from i3 import utils


class TestSafeLog(unittest.TestCase):
  """Test safe_log function."""

  def test_log_standard(self):
    """Test that safe_log does not differ from log for values > 0."""
    for num in [1, 10, 100, 1000]:
      self.assertAlmostEqual(utils.safe_log(num), math.log(num))

  def test_log_extended(self):
    """Test that safe_log returns -inf for 0."""
    self.assertEqual(utils.safe_log(0), utils.NEGATIVE_INFINITY)
    self.assertEqual(utils.NEGATIVE_INFINITY, float("-inf"))


if __name__ == '__main__':
  unittest.main()
