from __future__ import division

import os
import sqlalchemy as sa


def get_session(database_url, echo=False):
  """Return new SQLAlchemy session."""
  engine = sa.create_engine(database_url, echo=echo)
  sessionmaker = sa.orm.sessionmaker(bind=engine)
  return sessionmaker()


def get_database_url(database_file=None):
  """Read database information from .pgpass.

  pgpass format: ip:port:database:user:password
  url format: postgresql://{user}:{password}@{host}:{port}/{db_name}
  """
  if not database_file:
    home = os.path.expanduser("~")
    database_file = os.path.join(home, ".pgpass")
  host, port, db_name, user, password = open(database_file).read().split(":")
  return "postgresql://{user}:{password}@{host}:{port}/{db_name}".format(
    host=host,
    port=port,
    db_name=db_name,
    user=user,
    password=password
  )
  

def reset_database(sql_base_class, db_url):
  """Drop and recreate all tables."""
  engine = sa.create_engine(db_url, echo=False)
  sql_base_class.metadata.drop_all(bind=engine)
  sql_base_class.metadata.create_all(bind=engine)
  
