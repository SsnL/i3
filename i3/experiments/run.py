from __future__ import division
import os
import sys
sys.path.append(os.getcwd())

import cloud
import random
import sqlalchemy as sa
import time
import multiprocessing as mp
import sys

import i3
from i3.experiments import sql
from i3.experiments import experiment1


def run_job(job_id, url):
  experiment = experiment1
  """Run a single parameterized job stored in database for experiment."""

  def log(s):
      sys.stdout.write("{}\t{}\n".format(job.id, s))

  # Retrieve job from database
  max_wait_time = 10
  max_tries = 10
  num_tries = 0
  success = False
  while not success and num_tries < max_tries:
    try:
      session = sql.get_session(url)
      job = session.query(experiment.Job).filter(
        experiment.Job.id == job_id).first()
    except sa.exc.OperationalError:
      log("Could not reach database, retrying...")
      num_tries += 1
      time.sleep(random.random() * max_wait_time)
    else:
      success = True
      log("Successfully retrieved job.")

  if not success:
    raise Exception, "Maximum number of connection attempts exceeded."

  # Run job
  try:
    experiment.run(job, session, log)
  except Exception as e:
    job.status = "fail ({})".format(e)
  else:
    job.status = "done"
  finally:
    session.commit()
    session.close()

  return job.status

def run_experiment(experiment, reset_database=False):
  """Create and run all jobs for experiment."""
  url = sql.get_database_url()
  if reset_database:
    print "Resetting database..."
    sql.reset_database(experiment.Runs, url)
    sql.reset_database(experiment.DiscreteData, url)
    sql.reset_database(experiment.Job, url)
  print "Creating jobs..."
  # jobs = experiment.create_jobs(num_jobs=2)
  # jobs = experiment.create_reference_jobs(10)
  session = sql.get_session(url)
  jobs = session.query(experiment.Job).filter(experiment.Job.status != 'done')
  # session.add_all(jobs)
  # session.commit()
  job_ids = [job.id for job in jobs]
  session.close()
  print "Running jobs {}...".format(job_ids)
  p = mp.Pool()
  for job_id in job_ids:
    p.apply_async(run_job, args = (job_id, url))
  p.close()
  p.join()
  p.terminate()
  # run = lambda job_id: run_job(experiment, job_id, url)
  #   run(job_id)
  # cloud.map(run, job_ids, _type="f2")
  print "Done!"


if __name__ == "__main__":
  run_experiment(experiment1)
