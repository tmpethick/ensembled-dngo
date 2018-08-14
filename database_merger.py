import pandas as pd

import config as config
conf = config.get_config()

try:
  local = pd.read_csv(conf['database'])
  remote_temp_database = pd.read_csv(conf['remote_temp_database'])

  df = pd.concat([remote_temp_database, local], sort=False)
  df.groupby(["uuid"]).agg(lambda x: x.iloc[0])
except FileNotFoundError:
  df = pd.read_csv(conf['remote_temp_database'])

df.to_csv(conf['database'])
