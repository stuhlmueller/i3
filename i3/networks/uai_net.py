import cloud
import itertools
import os

from i3 import uai_import


def data_path():
  if cloud.running_on_cloud():
    return "/bucket/"
  else:
    return os.path.join(os.path.dirname(__file__), "../../data/")

    
def network_path():
  return os.path.join(data_path(), "networks/uai/")


def names(max_net_size):
  for filename in os.listdir(network_path()):
    f = open(os.path.join(network_path(), filename))
    line2 = list(itertools.islice(f, 2))[1]
    if int(line2) <= max_net_size:
      yield filename


def get(rng, name):
  filename = os.path.join(network_path(), name)
  net = uai_import.load_network(filename, rng)
  return net


def evidence(name):
  evidence_path = os.path.join(data_path(), "evidence/uai/")
  filename = os.path.join(evidence_path, "{}.evid".format(name))
  evidence = uai_import.load_evidence(filename)
  return evidence


def marginals(name):
  marginals_path = os.path.join(data_path(), "marginals/uai/")
  true_filename = os.path.join(marginals_path, "{}.true.mar".format(name))
  approx_filename = os.path.join(marginals_path, "{}.approx.mar".format(name))
  if os.path.exists(true_filename):
    marginals = uai_import.load_marginals(true_filename)
    marginals_exact = True
  else:
    marginals = uai_import.load_marginals(approx_filename)
    marginals_exact = False
  return marginals, marginals_exact
