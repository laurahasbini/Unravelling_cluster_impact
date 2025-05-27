import math
import os
import glob
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import xarray as xr 
import datetime 
import pytz
import csv
from scipy.stats import kde
from haversine import haversine
import netCDF4
import xskillscore as xs
from scipy import stats

from fct.paths import *

period         = "1979-2024WIN"

### MERGING THE INFO FILES 
csv_info_files = glob.glob(PATH_TRACKS+"/season/*_info.csv")

# Combine into a single DataFrame
dfs = [pd.read_csv(file) for file in csv_info_files]
combined_df_info = pd.concat(dfs, ignore_index=True)
combined_df_info = combined_df_info.sort_values('storm_landing_date')

# Save to a new CSV file
combined_df_info.to_csv(PATH_TRACKS+"tracks_"+filt+"_"+period+"_info.csv", index=False)

## MERGING THE TRACKS FILES 
file_pattern   = os.path.join(PATH_TRACKS+"season/", "tracks_"+filt+"_*.csv")  

# Find and filter the matching files
csv_files      = [f for f in glob.glob(file_pattern) if not f.endswith("_info.csv")]

# Ensure the files are sorted for proper ordering
csv_files.sort()

# Combine all files into one DataFrame
combined_df    = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
combined_df    = combined_df.sort_values(['year', 'month', 'day', 'hour'])

# Save the merged DataFrame to a new CSV file
combined_df.to_csv(PATH_TRACKS+"tracks_"+filt+"_"+period+".csv", index=False)

###### REMOVE TRACKS APRIL/MAY 
combined_df_info['storm_landing_date'] = pd.to_datetime(combined_df_info['storm_landing_date'])
combined_df_info = combined_df_info.loc[combined_df_info.storm_landing_date.dt.month.isin([1, 2, 3, 9, 10, 11, 12])]
combined_df = combined_df.loc[combined_df.storm_id.isin(combined_df_info.storm_id)]

###### FILTER TRACKS MORE THAN 24h 
filtered_df = combined_df.groupby('storm_id').filter(lambda x: len(x) >= 4)
filtered_df_info = combined_df_info.loc[combined_df_info.storm_id.isin(filtered_df.storm_id.unique())]

filtered_df_info.to_csv(PATH_TRACKS+"tracks_ALL"+filt+"_24h_"+period+"_info.csv", index=False)
filtered_df.to_csv(PATH_TRACKS+"tracks_ALL"+filt+"_24h_"+period+".csv", index=False)

