
from i3 import bayesnet
from i3 import dist
from i3 import random_world
from i3 import utils


class TestBrightnessContrastBayesNet(object):
  #     (with-query ((reflectance (gaussian 1 1)))
  #         reflectance
  #         (= (gaussian (* reflectance (+ (gamma 9 0.5) (gaussian 1 0.5))) 2) 10))

  def setup(self):
    self.rng = utils.RandomState(seed=0)
    self.reflectance = bayesnet.RealBayesNetNode(
        index=0, dist_fn=lambda _: dist.GaussianDistribution(self.rng, 1, 1))
    self.illumination_1 = bayesnet.RealBayesNetNode(
        index=1, dist_fn=lambda _: dist.GammaDistribution(self.rng, 9, 0.5))
    self.illumination_2 = bayesnet.RealBayesNetNode(
        index=2, dist_fn=lambda _: dist.GaussianDistribution(self.rng, 1, 0.5))
    self.illumination = bayesnet.DeterministicBayesNetNode(
        index=3, function=lambda parents: parents[0] + parents[1],
        rng=self.rng)
    self.observation = bayesnet.RealBayesNetNode(
        index=4, dist_fn=lambda parents: dist.GaussianDistribution(self.rng, parents[0] * parents[1], 2))
    self.net = bayesnet.BayesNet(self.rng,
        nodes=[self.reflectance, self.illumination_1, self.illumination_2, self.illumination, self.observation],
        edges=[(self.reflectance, self.observation),
               (self.illumination_1, self.illumination),
               (self.illumination_2, self.illumination),
               (self.illumination, self.observation)])
    self.net.compile()
    self.inverse_map = invert.compute_inverse_map(
      self.net, [self.observation], self.rng)

  def test_samples(self):
    worlds = []
    for i in range(10):
      worlds.append(self.net.sample())
    print worlds

  def test_inverse_inferences(self):
    trainer = train.Trainer(self.inverse_map)
    for _ in xrange(500):
      world = self.net.sample()
      trainer.observe(world)
    trainer.finalize()


if __name__ == '__main__':
  net = TestBrightnessContrastBayesNet()
  net.setup()
  net.test_samples()

