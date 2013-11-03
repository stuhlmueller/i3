import numpy as np

from i3 import bayesnet
from i3 import dist
from i3 import evid
from i3 import invert
from i3 import mcmc
from i3 import train
from i3 import utils


class TestBrightnessContrastBayesNet(object):
  # (with-query ((reflectance (gaussian 1 1)))
  #     reflectance
  #     (= (gaussian (* reflectance (+ (gamma 9 0.5) (gaussian 1 0.5))) 2) 10))

  def setup(self):
    self.rng = utils.RandomState(seed=0)
    self.reflectance = bayesnet.DistRealBayesNetNode(
      index=0, distribution=dist.GaussianDistribution(self.rng, 1, 1))
    self.illumination_1 = bayesnet.DistRealBayesNetNode(
      index=1, distribution=dist.GammaDistribution(self.rng, 9, 0.5))
    self.illumination_2 = bayesnet.DistRealBayesNetNode(
      index=2, distribution=dist.GaussianDistribution(self.rng, 1, 0.5))
    def observation_func(parents):
      return dist.GaussianDistribution(
        self.rng, parents[0] * (parents[1] + parents[2]), 2)
    self.observation = bayesnet.DistRealBayesNetNode(
      index=3, distribution=dist.FunctionDistribution(self.rng, observation_func))
    self.net = bayesnet.BayesNet(self.rng,
        nodes=[self.reflectance, self.illumination_1,
               self.illumination_2, self.observation],
        edges=[(self.reflectance, self.observation),
               (self.illumination_1, self.observation),
               (self.illumination_2, self.observation)])
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
    trainer = train.Trainer(self.net, inverse_map, False, k=50)
    for _ in xrange(1000):
      world = self.net.sample()
      trainer.observe(world)
    trainer.finalize()
    evidence = evid.Evidence(keys=[self.observation], values=[5.5])
    proposal_size = 4
    num_samples = 500
    test_sampler = mcmc.InverseChain(
      self.net, inverse_map, self.rng, evidence, proposal_size)
    test_sampler.initialize_state()
    states = []
    for i in range(num_samples):
      test_sampler.transition()
      states.append(test_sampler.state)
      if i % 10 == 0:
        print states[-1]
    print np.mean([s[0] for s in states])


if __name__ == '__main__':
  net = TestBrightnessContrastBayesNet()
  net.setup()
  # net.test_samples()
  net.test_inverse_inferences()

