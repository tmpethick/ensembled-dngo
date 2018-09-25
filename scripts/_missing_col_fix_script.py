import pandas as pd
import numpy as np
from ast import literal_eval

from dateutil import parser

import config
conf = config.get_config()
df = pd.read_csv(conf['database'])
df['seed'] = pd.Series(None, index=df.index)
df['n_init'] = pd.Series(2, index=df.index)
df.date = df.date.astype('datetime64[ns]')

# TODO: Delete the seedless
df = df.drop(df[(df.date < parser.parse("2018-08-19 15:00")) & (df.date > parser.parse("2018-08-19 13:00"))].index, axis=0)

df.seed = df.seed.fillna(-1).astype('int64')
df.n_init = df.n_init.astype('int64')

# epoch
df.loc[(df.date > parser.parse("2018-08-19 15:00")), 'seed'] = 1
df.loc[(df.date > parser.parse("2018-08-19 15:00")), 'n_init'] = 20

df['embedding'] = pd.Series('None', index=df.index)
df.loc[(df.obj_func == 'sinone') & (df.date > parser.parse("2018-08-19 15:00")), 'embedding'] = '[0]'
df.loc[(df.obj_func == 'branin') & (df.date > parser.parse("2018-08-19 15:00")), 'embedding'] = '[0]'

df['embedding'] = df.embedding.fillna('None').apply(literal_eval)

df.to_csv(conf['database'], index=False)
