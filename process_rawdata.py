import pandas as pd
import datetime
import numpy as np
import h5py
from fuel.datasets import H5PYDataset
from config import config
import os
import glob
import geohash

locals().update(config)
network_mode = 0

date_parser = pd.tseries.tools.to_datetime

path = './data/workspace/'  # use your path
all_files = glob.glob(os.path.join(path, "*.csv"))
print("total number of datasets: " + str(len(all_files)) + ", in " + str(len(all_files) / 2) + " months")

df_each = (pd.read_csv(f, index_col=False, header=0, usecols=[2, 7, 8], parse_dates=[0], infer_datetime_format=True,
                       names=['Dropoff_datetime', 'Dropoff_longitude', 'Dropoff_latitude'], memory_map=True) for f in
           all_files)

df = pd.concat(df_each, ignore_index=True)
print(df.shape)
df = df[(df['Pickup_longitude'] > -74.06) & (df['Pickup_longitude'] < -73.77) &
        (df['Pickup_latitude'] < 40.91) & (df['Pickup_latitude'] > 40.61)]
print(df.shape)

df['Pickup_datetime'] = df['Pickup_datetime'].apply(lambda x: x.replace(minute=(x.minute / 5) * 5, second=0))
print(df[:10])
df['Pickup_geohash'] = df[['Pickup_latitude', 'Pickup_longitude']].apply(
    lambda x: geohash.encode(x[0], x[1], precision=7), axis=1)
print(df[:10])
df = df[['Pickup_datetime', 'Pickup_geohash']]
print(df.head(10))
print(df.shape)
df.to_csv('./output/2016_5_second.csv', index=False)
