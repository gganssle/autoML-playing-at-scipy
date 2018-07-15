import dask.dataframe as dd
from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler
from dask.diagnostics import visualize
from dask.distributed import Client

import pandas as pd

client = Client()

pandas_df = pd.read_csv('CurrentOrders201805301800.csv')

pandas_df.columns

pandas_df = pandas_df.fillna(value=0)
pandas_df.to_csv('no_nans.csv')

df = dd.read_csv('no_nans.csv', dtype={'Unnamed: 24': 'object'})

df.head()
df.groupby(df['35.1822']).mean().compute()

with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
     out = df.groupby(df['35.1822']).mean().compute()

prof.visualize()
visualize([prof, rprof, cprof])

import dask.array as da
a = da.random.random(size=(10000, 10000), chunks=(1000, 10000))
q, r = da.linalg.qr(a)
a2 = q.dot(r)

with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
     out = a2.compute()
