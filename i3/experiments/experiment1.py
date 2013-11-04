# For triangle network, check dependence of performance on parameter settings.
from __future__ import division

import datetime
import random
import sqlalchemy as sa
from sqlalchemy.ext import declarative as sa_declarative

from i3 import invert
from i3 import learn
from i3 import marg
from i3 import mcmc
from i3 import train
from i3 import utils
from i3.networks import triangle_net


SQLBase = sa_declarative.declarative_base()


class Job(SQLBase):
  __tablename__ = "experiment1"

  id = sa.Column(sa.Integer, primary_key=True)
  name = sa.Column(sa.String)
  status = sa.Column(sa.String)
  determinism = sa.Column(sa.Integer)
  inversion_seconds = sa.Column(sa.Float)
  learner = sa.Column(sa.String)
  max_inverse_size = sa.Column(sa.Integer)
  num_training_samples = sa.Column(sa.Integer)
  precompute_gibbs = sa.Column(sa.Boolean)
  seed = sa.Column(sa.Integer)
  start_time = sa.Column(sa.DateTime)
  test_error = sa.Column(sa.Float)
  test_proposals = sa.Column(sa.Integer)
  test_proposals_accepted = sa.Column(sa.Integer)
  test_seconds = sa.Column(sa.Float)
  empirical_test_seconds = sa.Column(sa.Float)  
  training_error = sa.Column(sa.Float)
  training_seconds = sa.Column(sa.Float)
  training_source = sa.Column(sa.String)
  integrated_error = sa.Column(sa.Float)

  def __init__(self, name):
    self.name = name
    self.status = "init"
    self.determinism = 95
    self.learner = "counts"
    self.max_inverse_size = 1
    self.num_training_samples = 10000
    self.precompute_gibbs = False
    self.seed = 0
    self.test_seconds = 10
    self.training_source = "gibbs"
    self.inversion_seconds = None
    self.start_time = None
    self.test_error = None
    self.test_proposals = None
    self.test_proposals_accepted = None
    self.training_error = None
    self.training_seconds = None
    self.empirical_test_seconds = None
    self.integrated_error = None
    
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


def run(job, session):

  print "Starting job..."
  job.start_time = datetime.datetime.now()
  rng = utils.RandomState(job.seed)
  net = triangle_net.get(rng, job.determinism)
  evidence = triangle_net.evidence(0, job.determinism)
  evidence_nodes = [net.nodes_by_index[index] for index in evidence.keys()]
  num_latent_nodes = len(net.nodes()) - len(evidence_nodes)
  marginals = triangle_net.marginals(0, job.determinism)
  job.status = "started"
  session.commit()

  print "Computing inverse map..."
  t0 = datetime.datetime.now()
  inverse_map = invert.compute_inverse_map(
    net, evidence_nodes, rng, job.max_inverse_size)
  t1 = datetime.datetime.now()
  job.inversion_seconds = (t1 - t0).total_seconds()
  job.status = "inverted"
  session.commit()

  print "Training inverses..."
  if job.learner == "counts":
    learner_class = learn.CountLearner
  elif job.learner == "lr":
    learner_class = learn.LogisticRegressionLearner
  else:
    raise ValueError("Unknown learner type!")
  trainer = train.Trainer(net, inverse_map, job.precompute_gibbs, learner_class)
  counter = marg.MarginalCounter(net)
  if job.training_source in ("gibbs", "prior+gibbs"):
    training_sampler = mcmc.GibbsChain(net, rng, evidence)
    training_sampler.initialize_state()
    for _ in xrange(job.num_training_samples):
      training_sampler.transition()
      trainer.observe(training_sampler.state)
      counter.observe(training_sampler.state)
  if job.training_source in ("prior", "prior+gibbs"):
    for _ in xrange(job.num_training_samples):
      world = net.sample()
      trainer.observe(world)
      counter.observe(world)
  trainer.finalize()
  job.training_error = (marginals - counter.marginals()).mean()
  t2 = datetime.datetime.now()
  job.training_seconds = (t2 - t1).total_seconds()
  job.status = "trained"
  session.commit()

  print "Testing inverse sampler..."
  test_sampler = mcmc.InverseChain(
    net, inverse_map, rng, evidence, job.max_inverse_size)
  test_sampler.initialize_state()
  counter = marg.MarginalCounter(net)
  num_proposals_accepted = 0
  test_start_time = datetime.datetime.now()
  i = 0
  error_integrator = utils.TemporalIntegrator()
  while ((datetime.datetime.now() - test_start_time).total_seconds()
         < job.test_seconds):
    accept = test_sampler.transition()
    counter.observe(test_sampler.state)
    num_proposals_accepted += accept
    i += 1
    if i % 100 == 0:
      error = (marginals - counter.marginals()).mean()
      error_integrator.observe(error)
  final_error = (marginals - counter.marginals()).mean()
  final_time = datetime.datetime.now()
  empirical_test_seconds = (final_time - test_start_time).total_seconds()
  error_integrator.observe(final_error)
  job.test_error = final_error
  job.integrated_error = error_integrator.integral / empirical_test_seconds
  job.test_proposals = i * num_latent_nodes
  job.test_proposals_accepted = num_proposals_accepted
  job.empirical_test_seconds = empirical_test_seconds
