import os

from i3 import uai_import

data_path = os.path.join(os.path.dirname(__file__), "../../data/")

def get(rng):
  filename = os.path.join(data_path, "networks/triangle-n120-s95.uai")
  net = uai_import.load_network(filename, rng)
  return net

def evidence():
  filename = os.path.join(data_path, "evidence/triangle-n120-s95-1.evid")
  evidence = uai_import.load_evidence(filename)
  assert len(evidence) == 1
  return evidence[0]

def marginals():
  filename = os.path.join(data_path, "marginals/triangle-n120-s95-1.mar")
  marginals = uai_import.load_marginals(filename)
  return marginals

