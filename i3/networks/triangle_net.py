import os

from i3 import uai_import


def get(rng):
  filename = os.path.join(
    os.path.dirname(__file__),
    "../../data/networks/triangle-n120-s95.uai")
  net = uai_import.load_network(filename, rng)
  return net
