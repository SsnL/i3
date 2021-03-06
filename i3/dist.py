"""Probability distributions."""
from __future__ import division

import collections
import scipy.stats.distributions as dists

from i3 import utils


class Distribution(object):
  """A probability distribution (sampler and scorer)."""

  def __init__(self, rng):
    self.rng = rng

  def sample(self, params):
    """Get a sample from the distribution."""
    raise NotImplementedError("sample")

  def log_probability(self, params, value):
    """Get the log probability of a value."""
    raise NotImplementedError("probability")


class FunctionDistribution(Distribution):
  """A conditional distribution formed by calling a function on the parameters
     to produce the conditional distribution."""

  def __init__(self, rng, function):
    super(FunctionDistribution, self).__init__(rng)
    self.function = function

  def sample(self, params):
    return self.function(params).sample(None)

  def log_probability(self, params, value):
    return self.function(params).log_probability(None, value)


class DiscreteDistribution(Distribution):
  """A discrete probability distribution (sampler, scorer, and support)."""

  def support(self, params):
    raise NotImplementedError("support")


class CategoricalDistribution(DiscreteDistribution):
  """A distribution over a finite number of values."""

  def __init__(self, values, probabilities, rng):
    """Create a categorical distribution.

    Args:
      values: an iterable of associated values
      probabilities: an iterable of probabilites
    """
    super(CategoricalDistribution, self).__init__(rng)
    self.support_values = values
    self.sampler = None
    self.value_to_logprob = None
    self.probabilities = utils.normalize(probabilities)
    self.compile()

  def compile(self):
    self.sampler = self.rng.categorical_sampler(
      self.support_values, self.probabilities)
    self.value_to_logprob = collections.defaultdict(
      lambda: utils.LOG_PROB_0)
    for value, prob in zip(self.support_values, self.probabilities):
      self.value_to_logprob[value] = utils.safe_log(prob)

  def sample(self, params):
    """Sample a single value from the distribution."""
    assert not params
    return self.sampler()

  def log_probability(self, params, value):
    """Return the log probability of a given value."""
    return self.value_to_logprob[value]

  def support(self, params):
    assert not params
    return self.support_values


class ContinuousDistribution(Distribution):
  pass


class GaussianDistribution(ContinuousDistribution):

  def __init__(self, rng, mean, stdev):
    super(GaussianDistribution, self).__init__(rng)
    self.mean = mean
    self.stdev = stdev

  def sample(self, params):
    assert not params
    return self.rng.normal(self.mean, self.stdev)

  def log_probability(self, params, value):
    assert not params
    return dists.norm.logpdf(value, self.mean, self.stdev)


class GammaDistribution(ContinuousDistribution):

  def __init__(self, rng, shape, scale):
    super(GammaDistribution, self).__init__(rng)
    self.shape = shape
    self.scale = scale

  def sample(self, params):
    assert not params
    return self.rng.gamma(self.shape, scale=self.scale)

  def log_probability(self, params, value):
    assert not params
    return dists.gamma.logpdf(value, self.shape, scale=self.scale)


