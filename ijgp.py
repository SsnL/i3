# Compute all marginals using ijgp.

import i3
from i3.networks import uai_net

import datetime
import os
import sys

ijgp_url = "http://www.hlt.utdallas.edu/~vgogate/ijgp"


def run_job(name):
  max_seconds = 240
  network_path = "i3/data/networks/uai/{}".format(name)
  evidence_path = "i3/data/evidence/uai/{}.evid".format(name)
  os.system("wget {}".format(ijgp_url))
  os.system("chmod +x ./ijgp")
  command = "perl -e 'alarm shift @ARGV; exec @ARGV' {} ".format(max_seconds)
  command += "'./ijgp {} {}'".format(network_path, evidence_path)
  start_time = datetime.datetime.now()
  os.system(command)
  total_time = (datetime.datetime.now() - start_time).total_seconds()
  is_exact = total_time < .9 * max_seconds
  return name, open("{}.MAR".format(name)).read(), is_exact


def main(name):
  print "running IJGP on {}".format(name)
  name, marginals_string, is_exact = run_job(name)
  exact = "true" if is_exact else "approx"
  f = open(
    os.path.join(
      os.path.dirname(__file__),
      "i3/data/marginals/uai/{}.{}.mar".format(name, exact)), "w")
  f.write(marginals_string)
  f.close()


if __name__ == "__main__":
  main(sys.argv[1])

