import os

from i3 import uai_import

data_path = os.path.join(os.path.dirname(__file__), "../../data/")


def get(rng, smooth=95):
  filename = os.path.join(
    data_path, "networks/triangle-n120-s{}.uai".format(smooth))
  net = uai_import.load_network(filename, rng)
  return net


def evidence(index, smooth=95):
  assert index == 0
  filename = os.path.join(
    data_path, "evidence/triangle-n120-s{}-1.evid".format(smooth))
  evidence = uai_import.load_evidence(filename)
  assert len(evidence) == 1
  return evidence[0]


def marginals(index, smooth=95):
  assert index == 0
  filename = os.path.join(
    data_path, "marginals/triangle-n120-s{}-1.mar".format(smooth))
  marginals = uai_import.load_marginals(filename)
  return marginals

