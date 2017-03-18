# For triangle network, check dependence of performance on parameter settings.
from __future__ import division

import datetime
import random
import sqlalchemy as sa
from sqlalchemy.ext import declarative as sa_declarative
from sqlalchemy.dialects import postgresql as sa_postgresql
from  sqlalchemy.sql.expression import func
from itertools import product
import sys
import multiprocessing as mp

if sys.version_info < (3, 0):
  from itertools import izip as zip

import numpy as np

from i3 import invert
from i3 import learn
from i3 import marg
from i3 import mcmc
from i3 import train
from i3 import utils
from i3 import random_world
from i3.networks import triangle_net
from i3.experiments import sql

SQLBase = sa_declarative.declarative_base()

# traning data
# assumes binary valued world, determinism = 0.99
# Gibbs runs
class Run(SQLBase):
  __tablename__ = "experiment1_gibbs_runs"
  id = sa.Column(sa.Integer, primary_key=True)
  net_name = sa.Column(sa.String)
  start_time = sa.Column(sa.DateTime)
  num_states = sa.Column(sa.Integer)
  evidence_indices = sa.Column(sa_postgresql.ARRAY(sa.Integer))
  evidence_values = sa.Column(sa_postgresql.ARRAY(sa.Integer))
  seed = sa.Column(sa.Integer)

  def __init__(self, net_name, num_states, evidence_indices, evidence_values, seed = 0):
    self.net_name = net_name
    self.start_time = datetime.datetime.now()
    self.num_states = num_states
    self.evidence_indices = evidence_indices
    self.evidence_values = evidence_values
    self.seed = seed

class DiscreteData(SQLBase):
  __tablename__ = "experiment1_gibbs_data"

  gibbs_id = sa.Column(sa.Integer, primary_key=True)
  state_id = sa.Column(sa.Integer, primary_key=True, nullable = False)
  time = sa.Column(sa.DateTime)
  world_indices = sa.Column(sa_postgresql.ARRAY(sa.Integer))
  world_values = sa.Column(sa_postgresql.ARRAY(sa.Integer))

  def __init__(self, gibbs_id, state_id, world_indices, world_values):
    self.gibbs_id = gibbs_id
    self.state_id = state_id
    self.time = datetime.datetime.now()
    self.world_indices = world_indices
    self.world_values = world_values

def gen_data_run(net_name, num_states, url, seed):
  session = sql.get_session(url)
  # random evidence
  evidence = triangle_net.evidence(net_name, 0, 99)
  np.random.seed(seed)
  for k in evidence:
    evidence[k] = np.random.choice([0, 1])
  run = Run(net_name, num_states, list(evidence.keys()), list(evidence.values()), seed)
  session.add(run)
  session.commit()
  rng = utils.RandomState(seed)
  net = triangle_net.get(net_name, rng, 99)
  training_sampler = mcmc.GibbsChain(net, rng, evidence)
  training_sampler.initialize_state()
  for i in xrange(num_states):
    training_sampler.transition()
    state = training_sampler.state
    state = DiscreteData(run.id, i + 1, list(state.keys()), list(state.values()))
    session.add(state)
    session.commit()
  session.commit()
  session.close()

def gen_data(net_name, num_gibbs_runs, num_states_per_run, seed = 1000):
  print "Gen data..."
  url = sql.get_database_url()
  p = mp.Pool()
  for _ in xrange(num_gibbs_runs):
    p.apply_async(gen_data_run, args = (net_name, num_states_per_run, url, seed))
    seed += 1
  p.close()
  p.join()
  p.terminate()

class Job(SQLBase):
  __tablename__ = "experiment1"

  id = sa.Column(sa.Integer, primary_key=True)
  name = sa.Column(sa.String)
  net_name = sa.Column(sa.String)
  status = sa.Column(sa.String)
  determinism = sa.Column(sa.Integer)
  inversion_seconds = sa.Column(sa.Float)
  learner = sa.Column(sa.String)
  max_inverse_size = sa.Column(sa.Integer)
  num_training_samples_gibbs = sa.Column(sa.Integer)
  num_training_samples_prior = sa.Column(sa.Integer)
  precompute_gibbs = sa.Column(sa.Boolean)
  seed = sa.Column(sa.Integer)
  start_time = sa.Column(sa.DateTime)
  num_test_iterations = sa.Column(sa.Integer)
  # test_error = sa.Column(sa.Float)
  test_errors = sa.Column(sa_postgresql.ARRAY(sa.Float))
  test_proposals = sa.Column(sa.Integer)
  test_proposals_accepted = sa.Column(sa.Integer)
  # test_seconds = sa.Column(sa.Float)
  empirical_test_seconds = sa.Column(sa.Float)
  training_error = sa.Column(sa.Float)
  training_seconds = sa.Column(sa.Float)
  # training_source = sa.Column(sa.String)
  # integrated_error = sa.Column(sa.Float)

  def __init__(self, name):
    self.name = name
    self.status = "init"
    self.determinism = 95
    self.inversion_seconds = None
    self.learner = "counts"
    self.max_inverse_size = 1
    # self.num_training_samples = 10000
    self.num_training_samples_gibbs = 5000
    self.num_training_samples_prior = 5000
    self.precompute_gibbs = False
    self.seed = 0
    # self.training_source = "gibbs"
    self.start_time = None
    self.num_test_iterations = 100
    # self.test_error = None
    self.test_errors = None
    self.test_proposals = None
    self.test_proposals_accepted = None
    # self.test_seconds = 10
    self.empirical_test_seconds = None
    self.training_error = None
    self.training_seconds = None
    # self.integrated_error = None

  def __repr__(self):
    return "<Job({}, {}, {})>".format(self.name, self.id, self.status)

  @property
  def test_acceptance_rate(self):
    return self.test_proposals_accepted / self.test_proposals


def create_jobs(num_jobs):
  jobs = []
  seed = 1000
  for _ in xrange(num_jobs):
    seed += 1
    job = Job("exp1")
    job.seed = seed
    job.training_source = random.choice(["prior", "gibbs", "prior+gibbs"])
    job.determinism = random.choice([95, 99])
    job.max_inverse_size = random.choice(
      range(1, 20) + [20, 30, 40, 50, 60, 70, 80, 90, 100])
    job.num_training_samples = random.choice(
      [10, 100, 1000, 10000, 20000, 50000])
    job.precompute_gibbs = random.choice([True, False])
    job.learner = random.choice(["counts", "lr"])
    jobs.append(job)
  return jobs

def create_reference_jobs(num_jobs_per_case):
  jobs = []
  seed = 1000
  net_names = ["75-25-{}".format(i) for i in xrange(2, 3)]
  determinisms = [99]
  prior_ratios = [1.]
  max_inverse_sizes = [20]
  num_training_sampless = [10000, 100000]
  precompute_gibbss = [True]
  learners = ["counts", "lr"]
  num_test_iterations = 1000
  params = [net_names, determinisms, prior_ratios, max_inverse_sizes, num_training_sampless, precompute_gibbss, learners]
  for net_name, determinism, prior_ratio, max_inverse_size, num_training_samples, precompute_gibbs, learner in product(*params):
    for _ in xrange(num_jobs_per_case):
      seed += 1
      job = Job("exp1")
      job.net_name = net_name
      job.seed = seed
      job.num_training_samples_prior = int(prior_ratio * num_training_samples)
      job.num_training_samples_gibbs = num_training_samples - job.num_training_samples_prior
      job.determinism = determinism
      job.max_inverse_size = max_inverse_size
      job.precompute_gibbs = precompute_gibbs
      job.learner = learner
      job.num_test_iterations = num_test_iterations
      jobs.append(job)
  return jobs

def run(job, session, log):

  log("Starting job...")
  job.start_time = datetime.datetime.now()
  rng = utils.RandomState(job.seed)
  np.random.seed(job.seed)
  net = triangle_net.get(job.net_name, rng, job.determinism)
  evidence = triangle_net.evidence(job.net_name, 0, job.determinism)
  evidence_nodes = [net.nodes_by_index[index] for index in evidence.keys()]
  num_latent_nodes = len(net.nodes()) - len(evidence_nodes)
  marginals = triangle_net.marginals(job.net_name, 0, job.determinism)
  job.status = "started"
  session.commit()

  log("Computing inverse map...")
  t0 = datetime.datetime.now()
  inverse_map = invert.compute_inverse_map(
    net, evidence_nodes, rng, job.max_inverse_size)
  t1 = datetime.datetime.now()
  job.inversion_seconds = (t1 - t0).total_seconds()
  job.status = "inverted"
  session.commit()

  log("Training inverses...")
  if job.learner == "counts":
    learner_class = learn.CountLearner
  elif job.learner == "lr":
    learner_class = learn.LogisticRegressionLearner
  else:
    raise ValueError("Unknown learner type!")
  trainer = train.Trainer(net, inverse_map, job.precompute_gibbs, learner_class = learner_class)
  counter = marg.MarginalCounter(net)
  if job.num_training_samples_gibbs > 0:
    for state in session.query(DiscreteData).\
                  order_by(func.random()).\
                  limit(job.num_training_samples_gibbs):
      world = random_world.RandomWorld(state.world_indices, state.world_values)
      trainer.observe(world)
      counter.observe(world)
    # training_sampler = mcmc.GibbsChain(net, rng, evidence)
    # training_sampler.initialize_state()
    # for _ in xrange(job.num_training_samples_gibbs):
    #   training_sampler.transition()
    #   trainer.observe(training_sampler.state)
    #   counter.observe(training_sampler.state)
  if job.num_training_samples_prior > 0:
    for _ in xrange(job.num_training_samples_prior):
      world = net.sample()
      trainer.observe(world)
      counter.observe(world)
      del world
  trainer.finalize()
  job.training_error = (marginals - counter.marginals()).mean()
  t2 = datetime.datetime.now()
  job.training_seconds = (t2 - t1).total_seconds()
  job.status = "trained"
  session.commit()

  log("Testing inverse sampler...")
  test_sampler = mcmc.InverseChain(
    net, inverse_map, rng, evidence, job.max_inverse_size)
  test_sampler.initialize_state()
  counter = marg.MarginalCounter(net)
  num_proposals_accepted = 0
  test_start_time = datetime.datetime.now()
  i = 0
  test_errors = []
  # error_integrator = utils.TemporalIntegrator()
  # while ((datetime.datetime.now() - test_start_time).total_seconds()
  #        < job.test_seconds):
  while i < job.num_test_iterations:
    accept = test_sampler.transition()
    counter.observe(test_sampler.state)
    num_proposals_accepted += accept
    i += 1
    test_errors.append((marginals - counter.marginals()).mean())
    # if i % 100 == 0:
    #   error = (marginals - counter.marginals()).mean()
    #   error_integrator.observe(error)
  # final_error = (marginals - counter.marginals()).mean()
  final_time = datetime.datetime.now()
  empirical_test_seconds = (final_time - test_start_time).total_seconds()
  # error_integrator.observe(final_error)
  job.test_errors = test_errors
  # job.test_error = final_error
  # job.integrated_error = error_integrator.integral / empirical_test_seconds
  job.test_proposals = i * num_latent_nodes
  job.test_proposals_accepted = num_proposals_accepted
  job.empirical_test_seconds = empirical_test_seconds
