import cloud
import os

from i3 import uai_import


def data_path():
  if cloud.running_on_cloud():
    return "/bucket/"
  else:
    return os.path.join(os.path.dirname(__file__), "../../data/")


def get(rng, determinism=95):
  filename = os.path.join(
    data_path(), "networks/triangle-n120-s{}.uai".format(determinism))
  net = uai_import.load_network(filename, rng)
  return net


def evidence(index, determinism=95):
  assert index == 0
  filename = os.path.join(
    data_path(), "evidence/triangle-n120-s{}-1.evid".format(determinism))
  evidence = uai_import.load_evidence(filename)
  return evidence


def marginals(index, determinism=95):
  assert index == 0
  filename = os.path.join(
    data_path(), "marginals/triangle-n120-s{}-1.mar".format(determinism))
  marginals = uai_import.load_marginals(filename)
  return marginals

