#!/usr/bin/env python

"""Test probability distributions."""

import math
import numpy as np

from i3 import distribution
from i3 import utils


def test_categorical_distribution():
  """Test categorical distribution."""
  rng = utils.RandomState(0)
  dist = distribution.CategoricalDistribution(
    values=["a", "b"],
    probabilities=[.3, .7],
    rng=rng)
  samples = [dist.sample() for _ in range(10000)]
  assert 2000 < samples.count("a") < 4000
  assert 6000 < samples.count("b") < 8000
  np.testing.assert_almost_equal(
    .3, math.exp(dist.log_probability("a")))
  np.testing.assert_almost_equal(
    .7, math.exp(dist.log_probability("b")))  
  assert sorted(dist.support()) == ["a", "b"]
