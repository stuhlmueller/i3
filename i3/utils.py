#!/usr/bin/env python

"""Utilities for stochastic sampling and probability calculations."""

import math


NEGATIVE_INFINITY = float('-inf')

LOG_PROB_0 = NEGATIVE_INFINITY

LOG_PROB_1 = 0.0


def safe_log(num):
  """Like math.log, but returns -infinity on 0."""
  if num == 0.0:
    return NEGATIVE_INFINITY
  return math.log(num)
