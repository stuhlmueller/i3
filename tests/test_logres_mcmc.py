# Compare MCMC using LR and Counts trainer on UAI network

import cloud
import datetime
import pytest

from i3 import invert
from i3 import learn
from i3 import marg
from i3 import mcmc
from i3 import random_world
from i3 import train
from i3 import utils
from i3.networks import triangle_net


@pytest.mark.slow
def test_logistic_regression_mcmc(learner_class_index=0, seed=0):
  max_inverse_size = 30
  train_seconds = 2*60
  test_seconds = 60
  
  rng = utils.RandomState(seed=seed)
  net = triangle_net.get(rng)
  evidence = triangle_net.evidence(0)
  marginals = triangle_net.marginals(0)
  evidence_nodes = [net.nodes_by_index[index] for index in evidence.keys()]
  learner_classes = [
    lambda support, rng: learn.LogisticRegressionLearner(
      support, rng, transform_inputs=learn.identity_transformer),
    lambda support, rng: learn.LogisticRegressionLearner(
      support, rng, transform_inputs=learn.square_transformer),    
    learn.CountLearner]
  learner_class = learner_classes[learner_class_index]
  num_latent_nodes = len(net.nodes()) - len(evidence_nodes)
  
  print "Inverting network..."
  inverse_map = invert.compute_inverse_map(
    net, evidence_nodes, rng, max_inverse_size)

  train_start_time = datetime.datetime.now()
  print "Initializing trainer..."  
  trainer = train.Trainer(net, inverse_map, False, learner_class=learner_class)
  print "Training..."
  sample = random_world.RandomWorld()
  while ((datetime.datetime.now() - train_start_time).total_seconds()
         < train_seconds):  
    sample = net.sample(sample)  # Prior!
    trainer.observe(sample)
    sample.data = {}
  trainer.finalize()

  print "Testing..."
  test_sampler = mcmc.InverseChain(
    net, inverse_map, rng, evidence, proposal_size=max_inverse_size)  
  test_sampler.initialize_state()
  error_integrator = utils.TemporalIntegrator()
  test_start_time = datetime.datetime.now()
  counter = marg.MarginalCounter(net)
  i = 0
  num_proposals_accepted = 0
  while ((datetime.datetime.now() - test_start_time).total_seconds()
         < test_seconds):
    accept = test_sampler.transition()
    num_proposals_accepted += accept
    counter.observe(test_sampler.state)
    i += 1
    if i % 100 == 0:
      error = (marginals - counter.marginals()).mean()
      error_integrator.observe(error)
  final_time = datetime.datetime.now()      
  empirical_test_seconds = (final_time - test_start_time).total_seconds()      
  final_error = (marginals - counter.marginals()).mean()
  error_integrator.observe(final_error)
  num_proposals = i * num_latent_nodes
  return (num_proposals,
          num_proposals_accepted,
          error_integrator.integral / empirical_test_seconds,
          final_error)


def run(params):
  return test_logistic_regression_mcmc(*params)


def main():
  jobs = []
  for seed in [0, 1, 2, 3, 4]:
    for learner_class_index in [0, 1]:
      jobs.append((learner_class_index, seed))
  print "Scheduling jobs..."
  jids = cloud.map(run, jobs, _type="f2")
  print "Waiting for results..."
  results = cloud.result(jids)
  for job, result in zip(jobs, results):
    print job, result


if __name__ == "__main__":
  main()