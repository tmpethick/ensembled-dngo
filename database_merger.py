import pandas as pd

import config as config
conf = config.get_config()

try:
  local = pd.read_csv(conf['database'])
  remote_temp_database = pd.read_csv(conf['remote_temp_database'])

  # Add missing columns (in case a new parameter was introduced)
  remove_missing_columns = set(local.columns.values) - set(remote_temp_database.columns.values)
  for col in remove_missing_columns:
      remote_temp_database[col] = pd.Series(None, index=remote_temp_database.index)
  
  local_missing_columns = set(remote_temp_database.columns.values) - set(local.columns.values)
  for col in local_missing_columns:
      local[col] = pd.Series(None, index=local.index)

  df = pd.concat([remote_temp_database, local], sort=False)
  df = df.groupby(["uuid"]).agg(lambda x: x.iloc[0]).reset_index()
except FileNotFoundError:
  df = pd.read_csv(conf['remote_temp_database'])

df.to_csv(conf['database'], index=False)
