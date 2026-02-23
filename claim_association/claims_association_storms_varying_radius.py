## This script run the association method
import multiprocessing as mp
import sys 
import time
import math
import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import xarray as xr 
import datetime 
import pytz
import csv
import seaborn as sns
from scipy.stats import kde
from haversine import haversine
from cartopy.geodesic import Geodesic
import shapely.geometry as sgeom
import netCDF4
import xskillscore as xs
import geopandas as gpd
from scipy import stats
from scipy.signal import find_peaks
import rioxarray 

#Masking array
from rechunker import rechunk
from rasterio import features
from affine import Affine

### External functions 
import fct.fct_link_storm_claim as fct_link_storm_claim
from fct.paths import *

mp.set_start_method('fork', force=True)
input_year = 1997
end_year = 2024

############ Open the needed data and perform the association 
sinclim_version = 'v2.1'
period = '1979-2024WIN'
r = 1300

# Varying parameters 
method = "wgust"
nb_min_claims = 50
d_after = 3
d_before = 3

if sinclim_version == 'v1' :
    sinclim = pq.read_table(PATH_GENERALI+'sinclim_anom.parquet')
elif sinclim_version =='v2.1' : 
    sinclim = pq.read_table(PATH_GENERALI+'sinclim_v2.1_anom.parquet')

sinclim_pd      =  sinclim.to_pandas(date_as_object = True, safe = False)
sinclim_pd_sort = sinclim_pd.sort_values('dat_sin')

type_claim = 'tempete'
if type_claim == "tempete" :
    save_name = "sinclim_"+sinclim_version+"_storm"
elif type_claim == "inondation" :
    save_name = "sinclim_"+sinclim_version+"_flood"

### Filter the sinclim data
sinclim_sort_storm = fct_link_storm_claim.claims_preprocess(sinclim_pd_sort, type_claim)

### OPEN STORM TRAJECTORIES
track_source = 'priestley_ALL'
df_info_storm = pd.read_csv(PATH_TRACKS+"tracks_ALL_24h_"+period+"_info.csv", encoding='utf-8')
df_info_storm['storm_landing_date'] = pd.to_datetime(df_info_storm['storm_landing_date'])
df_info_storm = df_info_storm.sort_values('storm_landing_date')
df_storm = pd.read_csv(PATH_TRACKS+"tracks_ALL_24h_"+period+".csv", encoding='utf-8')

# Trajectories between the input period 

df_info_storm = df_info_storm.loc[df_info_storm.storm_landing_date < datetime.datetime(year=end_year+1, 
                                                                                                                     month=4, day=1, hour=0
                                                                                                                    )]
df_info_storm = df_info_storm.loc[df_info_storm.storm_landing_date >= datetime.datetime(year=input_year, 
                                                                                                                 month=10, day=1, hour=0
                                                                                                                )]
df_storm = df_storm.loc[df_storm.storm_id.isin(df_info_storm.storm_id.unique())]

###LINK STORM AND DATA 
delta_t_after = datetime.timedelta(hours=d_after * 24)
delta_t_before =  datetime.timedelta(hours=-d_before * 24)

# Select raw sinclim data 
sinclim_raw = sinclim_sort_storm.loc[sinclim_sort_storm.dat_sin < datetime.datetime(year=end_year+1, month=4, day=1, hour=0)+delta_t_after].copy()
sinclim_raw = sinclim_raw.loc[sinclim_raw.dat_sin >= datetime.datetime(year=input_year, month=10, day=1, hour=0)-delta_t_before]

#For the largest windows make the association from scratch
path_sinclim_storm = (
    PATH_GENERALI
    + "final_varying_radius/"
    + f"{save_name}_d-{d_before}_d+{d_after}_{track_source}_{input_year}-{end_year}.csv"
)

# Check if the file already exists
if os.path.exists(path_sinclim_storm):
    sinclim_storm = pd.read_csv(path_sinclim_storm)
else:
    sinclim_storm = fct_link_storm_claim.assoc_storm_candidates(
        sinclim_raw, d_before, d_after, df_storm, df_info_storm
    )
    sinclim_storm.to_csv(path_sinclim_storm, encoding="utf-8", index=False)

## ADD WGUST
path_sinclim_storm_wgust = (
    PATH_GENERALI
    + "final_varying_radius/"
    + f"{save_name}_d-{d_before}_d+{d_after}_{track_source}_{input_year}-{end_year}_r-varying.csv"
)
combined_dataset_storm = fct_link_storm_claim.import_wgust_footprint_varying_radius(sinclim_storm)
combined_dataset_storm = combined_dataset_storm.rename_vars({"max_fg10" : "max_wind_gust"})

if os.path.exists(path_sinclim_storm_wgust):
    sinclim_storm = pd.read_csv(path_sinclim_storm_wgust)
else : 
#     max_workers = min(32, os.cpu_count() * 4)
    max_workers = int(os.environ.get("PBS_NCPUS", mp.cpu_count()))
    sinclim_storm = fct_link_storm_claim.add_wgust_new_parallel(sinclim_storm, combined_dataset_storm, max_workers, False)
    sinclim_storm.to_csv(path_sinclim_storm_wgust , encoding='utf-8', index=False)

#### Link to Higest windgust at location 

# Make a copy and drop NA values
sinclim_copy_cp = sinclim_storm.copy()#.dropna(subset=['wgust_max'])
sinclim_unique_wgust = (
    sinclim_copy_cp.loc[
        sinclim_copy_cp.groupby('cod_sin')['wgust_max'].idxmax()
    ]
)

path_sinclim_storm_unique_wgust = (
    PATH_GENERALI
    + "final_varying_radius/"
    + f"{save_name}_d-{d_before}_d+{d_after}_unique-wgust_{track_source}_{input_year}-{end_year}_r-varying.csv"
)
sinclim_unique_wgust.to_csv(path_sinclim_storm_unique_wgust, encoding='utf-8', index=False)

### Apply gathering of number of claims
nb_min_claims_start = 10

sinclim_unique_wgust_min_claims = fct_link_storm_claim.gather_claims_storm_iteration(sinclim_unique_wgust, df_info_storm, nb_min_claims, 
                                                                   nb_min_claims_start, 10)

#Correct the final wgust 
sinclim_unique_wgust_min_claims = sinclim_unique_wgust_min_claims.drop('wgust_max', axis=1)
sinclim_unique_wgust_min_claims = fct_link_storm_claim.add_wgust_new_parallel(sinclim_unique_wgust_min_claims, combined_dataset_storm, max_workers, False)
sinclim_unique_wgust_min_claims = sinclim_unique_wgust_min_claims.dropna(subset=['wgust_max'])
# sinclim_copy_cp = sinclim_copy_cp.compute()


path_sinclim_storm_wgust_min_claims = (
    PATH_GENERALI
    + "final_varying_radius/"
    + f"{save_name}_d-{d_before}_d+{d_after}_unique-wgust_min{nb_min_claims}_{track_source}_{input_year}-{end_year}_r-varying.csv"
)
sinclim_unique_wgust_min_claims.to_csv(path_sinclim_storm_wgust_min_claims, encoding='utf-8', index=False)