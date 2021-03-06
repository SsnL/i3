"""Test probability distributions."""
import math

import pytest
import numpy as np

from i3 import dist
from i3 import utils


def test_discrete_distribution():
  """Elementary tests for discrete distribution class."""
  rng = utils.RandomState(0)
  distribution = dist.DiscreteDistribution(rng)
  with pytest.raises(NotImplementedError):
    distribution.sample([])
  with pytest.raises(NotImplementedError):
    distribution.log_probability([], None)
  with pytest.raises(NotImplementedError):
    distribution.support([])


def test_categorical_distribution():
  """Test categorical distribution."""
  rng = utils.RandomState(0)
  distribution = dist.CategoricalDistribution(
    values=["a", "b"],
    probabilities=[.3, .7],
    rng=rng)
  samples = [distribution.sample([]) for _ in range(10000)]
  utils.assert_in_interval(samples.count("a"), .3, 10000, .95)
  utils.assert_in_interval(samples.count("b"), .7, 10000, .95)
  np.testing.assert_almost_equal(
    .3, math.exp(distribution.log_probability([], "a")))
  np.testing.assert_almost_equal(
    .7, math.exp(distribution.log_probability([], "b")))
  assert sorted(distribution.support([])) == ["a", "b"]
