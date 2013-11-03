
from i3 import bayesnet
from i3 import dist
from i3 import evid
from i3 import invert
from i3 import mcmc
from i3 import random_world
from i3 import train
from i3 import utils

import math

def mean(items):
  return sum(items) / float(len(items))

class TestBrightnessContrastBayesNet(object):
  #     (with-query ((reflectance (gaussian 1 1)))
  #         reflectance
  #         (= (gaussian (* reflectance (+ (gamma 9 0.5) (gaussian 1 0.5))) 2) 10))

  def setup(self):
    self.rng = utils.RandomState(seed=0)
    self.reflectance = bayesnet.DistRealBayesNetNode(
        index=0, distribution=dist.GaussianDistribution(self.rng, 1, 1))
    self.illumination_1 = bayesnet.DistRealBayesNetNode(
        index=1, distribution=dist.GammaDistribution(self.rng, 9, 0.5))
    self.illumination_2 = bayesnet.DistRealBayesNetNode(
        index=2, distribution=dist.GaussianDistribution(self.rng, 1, 0.5))
    self.illumination = bayesnet.DistRealBayesNetNode(
        index=3, distribution=dist.FunctionDistribution(self.rng,
          lambda parents: dist.GaussianDistribution(self.rng, parents[0] + parents[1], 0.2)))
        rng=self.rng)
    self.observation = bayesnet.DistRealBayesNetNode(
        index=3, distribution=dist.FunctionDistribution(self.rng,
          lambda parents: dist.GaussianDistribution(self.rng, parents[0] * parents[1], 2)))
    self.net = bayesnet.BayesNet(self.rng,
        nodes=[self.reflectance, self.illumination_1, self.illumination_2, self.illumination, self.observation],
        edges=[(self.reflectance, self.observation),
               (self.illumination_1, self.illumination),
               (self.illumination_2, self.illumination),
               (self.illumination, self.observation)])
    self.net.compile()

  def test_samples(self):
    worlds = []
    for i in range(10):
      worlds.append(self.net.sample())
    print worlds

  def test_inverse_inferences(self):
    inverse_map = invert.compute_inverse_map(
      self.net, [self.observation], self.rng)
    print inverse_map.nets_by_key
    trueval = 1.09902618098
    # for trainsamps in [50, 100, 200, 500, 1000, 2000, 5000]:
    # for trainsamps in [10, 10000, 100000]:
    # for trainsamps in [20, 20000, 50000]:
    for trainsamps in [10, 100, 1000, 10000, 100000]:
      errors = []
      # for trial in range(50):
      for trial in range(1):
        k = math.sqrt(trainsamps)
        trainer = train.Trainer(self.net, inverse_map, False, k=k)
        for _ in xrange(trainsamps):
          world = self.net.sample()
          trainer.observe(world)
        trainer.finalize()

        evidence = evid.Evidence(keys=[self.observation], values=[5.5])
        proposal_size = 4
        num_samples = 1000
        test_sampler = mcmc.InverseChain(
          self.net, inverse_map, self.rng, evidence, proposal_size)
        test_sampler.initialize_state()
        states = []
        for i in range(num_samples):
          test_sampler.transition()
          states.append(test_sampler.state)
        m = mean([s[0] for s in states])
        print 'trainsamps %s trial %s mean %s' % (trainsamps, trial, m)
        errors.append(m-trueval)
      rmse = math.sqrt(mean([e**2 for e in errors]))
      print 'trainsamps %s rmse %s' % (trainsamps, rmse)



if __name__ == '__main__':
  net = TestBrightnessContrastBayesNet()
  net.setup()
  # net.test_samples()
  net.test_inverse_inferences()

