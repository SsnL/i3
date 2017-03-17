# import cloud
import os

from i3 import uai_import


def data_path():
  return os.path.join(os.path.dirname(__file__), "../../data/")


def get(name, rng, determinism=95):
  # filename = os.path.join(
  #   data_path(), "networks/triangle-n120-s{}.uai".format(determinism))
  filename = os.path.join(
    data_path(), "networks/{}-s{}.uai".format(name, determinism))
  net = uai_import.load_network(filename, rng)
  return net


def evidence(name, index, determinism=95):
  assert index == 0
  # filename = os.path.join(
  #   data_path(), "evidence/triangle-n120-s{}-1.evid".format(determinism))
  filename = os.path.join(
    data_path(), "evidence/{}-s{}-1.evid".format(name, determinism))
  evidence = uai_import.load_evidence(filename)
  return evidence


def marginals(name, index, determinism=95):
  assert index == 0
  # filename = os.path.join(
  #   data_path(), "marginals/triangle-n120-s{}-1.mar".format(determinism))
  filename = os.path.join(
    data_path(), "marginals/{}-s{}-1.mar".format(name, determinism))
  marginals = uai_import.load_marginals(filename)
  return marginals

