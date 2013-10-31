from __future__ import division

import cloud
import random
import sqlalchemy as sa
import time

from i3.experiments import sql
from i3.experiments import experiment1


def run_job(experiment, job_id, url):
  """Run a single parameterized job stored in database for experiment."""

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
      print "Could not reach database, retrying..."
      num_tries += 1
      time.sleep(random.random() * max_wait_time)
    else:
      success = True
      print "Successfully retrieved job."

  if not success:
    raise Exception, "Maximum number of connection attempts exceeded."

  # Run job
  try:
    experiment.run(job, session)
  except Exception, e:
    job.status = "fail ({})".format(e)
  else:
    job.status = "done"
  finally:
    session.commit()
    session.close()

  
def run_experiment(experiment, reset_database=False):
  """Create and run all jobs for experiment."""
  url = sql.get_database_url()
  if reset_database:
    print "Resetting database..."
    sql.reset_database(experiment.SQLBase, url)
  print "Creating jobs..."
  jobs = experiment.create_jobs(num_jobs=500)
  session = sql.get_session(url)
  session.add_all(jobs)
  session.commit()
  job_ids = [job.id for job in jobs]
  session.close()
  print "Running jobs..."
  run = lambda job_id: run_job(experiment, job_id, url)
  cloud.map(run, job_ids, _type="f2")
  print "Done!"


if __name__ == "__main__":
  run_experiment(experiment1)
