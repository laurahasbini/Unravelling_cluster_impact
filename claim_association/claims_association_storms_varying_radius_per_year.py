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
input_year = int(sys.argv[1])
end_year = input_year+1

############ Open the needed data and perform the association 
sinclim_version = "v2.2"#'v2.1'#'v2.2'
period = '1979-2024WIN'
r = 1300

# Varying parameters 
method        = "wgust"
nb_min_claims = 50
d_after       = 3
d_before      = 3
type_claim    = 'tempete' # tempete_extended

if sinclim_version == 'v1' :
    sinclim = pq.read_table(PATH_GENERALI+'sinclim_anom.parquet')
elif sinclim_version =='v2.1' : 
    sinclim = pq.read_table(PATH_GENERALI+'sinclim_v2.1_anom.parquet')
elif sinclim_version =='v2.2' : 
#     sinclim = pq.read_table(PATH_GENERALI+'sinclim_v2.2_anom_v031125.parquet')
    sinclim = pq.read_table(PATH_GENERALI+'sinclim_v2.2_anom_v101225.parquet')

sinclim_pd      =  sinclim.to_pandas(date_as_object = True, safe = False)
sinclim_pd_sort = sinclim_pd.sort_values('dat_sin')

if type_claim == "tempete" :
    save_name = "sinclim_"+sinclim_version+"_storm"
    lib_per_sel = [type_claim]
elif type_claim == "inondation" :
    save_name = "sinclim_"+sinclim_version+"_flood"
    lib_per_sel = [type_claim]
elif type_claim == "tempete_extended" :
    save_name = "sinclim_"+sinclim_version+"_storm_extended"
    lib_per_sel = ["tempete", "orage", "degat_des_eaux"]

### Filter the sinclim data
sinclim_sort_storm = fct_link_storm_claim.claims_preprocess(sinclim_pd_sort, lib_per_sel)

### OPEN STORM TRAJECTORIES
track_source = 'priestley_ALL'
df_info_storm = pd.read_csv(PATH_TRACKS+"tracks_FR_ALL_24h_"+period+"_info.csv", encoding='utf-8')
df_info_storm['storm_landing_date'] = pd.to_datetime(df_info_storm['storm_landing_date'])
df_info_storm = df_info_storm.sort_values('storm_landing_date')
df_storm = pd.read_csv(PATH_TRACKS+"tracks_FR_ALL_24h_"+period+".csv", encoding='utf-8')

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
    PATH_TEMP
    + "final_per_year/" 
    + f"{save_name}_d-{d_before}_d+{d_after}_{track_source}_{input_year}-{end_year}.csv"
)

## Check if the file already exists
if os.path.exists(path_sinclim_storm):
    sinclim_storm = pd.read_csv(path_sinclim_storm, dtype=fct_link_storm_claim.DTYPE_SINCLIM)    
else:
    sinclim_storm = fct_link_storm_claim.assoc_storm_candidates(
        sinclim_raw, d_before, d_after, df_storm, df_info_storm
    )
    sinclim_storm.to_csv(path_sinclim_storm, encoding="utf-8", index=False)

## ADD WGUST
path_sinclim_storm_wgust = (
    PATH_TEMP
    + "final_per_year/" 
    + "r-varying/"
    + f"{save_name}_d-{d_before}_d+{d_after}_{track_source}_{input_year}-{end_year}_r-varying.csv"
)
combined_dataset_storm = fct_link_storm_claim.import_wgust_footprint_varying_radius(sinclim_storm)
combined_dataset_storm = combined_dataset_storm.rename_vars({"max_fg10" : "max_wind_gust"})

if os.path.exists(path_sinclim_storm_wgust):
    sinclim_storm = pd.read_csv(path_sinclim_storm_wgust, dtype=fct_link_storm_claim.DTYPE_SINCLIM)    
else : 
    max_workers = int(os.environ.get("PBS_NCPUS", mp.cpu_count()))
    sinclim_storm = fct_link_storm_claim.add_wgust_new_parallel(sinclim_storm, combined_dataset_storm, max_workers, False)
    sinclim_storm.to_csv(path_sinclim_storm_wgust , encoding='utf-8', index=False)

#### Link to Higest windgust at location 

path_sinclim_storm_unique_wgust = (
    PATH_TEMP
    + "final_per_year/" 
    + "r-varying/"
    + f"{save_name}_d-{d_before}_d+{d_after}_unique-wgust_{track_source}_{input_year}-{end_year}_r-varying.csv"
)

if os.path.exists(path_sinclim_storm_unique_wgust):
    sinclim_unique_wgust = pd.read_csv(path_sinclim_storm_unique_wgust, dtype=fct_link_storm_claim.DTYPE_SINCLIM)
else : 
    sinclim_copy_cp = sinclim_storm.copy()
    idx_list = []

    for cod_sin, group in sinclim_copy_cp.groupby('cod_sin'):
        if group['wgust_max'].notna().any():
            idx = group['wgust_max'].idxmax()
        else:
            idx = group.index[0]  # take the first row if all NaN
        idx_list.append(idx)
    sinclim_unique_wgust = sinclim_copy_cp.loc[idx_list]

    #Save the new dataframe 
    sinclim_unique_wgust.to_csv(path_sinclim_storm_unique_wgust, encoding='utf-8', index=False)

### Apply gathering of number of claims
nb_min_claims_start = 10

sinclim_unique_wgust["storm_id_old"] = sinclim_unique_wgust["storm_id"]
sinclim_unique_wgust["wgust_max_old"] = sinclim_unique_wgust["wgust_max"]
sinclim_unique_wgust_min_claims = fct_link_storm_claim.gather_claims_storm_iteration(sinclim_unique_wgust, df_info_storm, nb_min_claims, 
                                                                   nb_min_claims_start, 10)

#Change the wgust for the claims which have been shifted 
mask_switched = sinclim_unique_wgust_min_claims["storm_id"] != sinclim_unique_wgust_min_claims["storm_id_old"]
claims_to_update = sinclim_unique_wgust_min_claims.loc[mask_switched].copy()

# Parallel recomputation only for those
max_workers = int(os.environ.get("PBS_NCPUS", mp.cpu_count()))
claims_to_update = fct_link_storm_claim.add_wgust_new_parallel(
    claims_to_update,
    combined_dataset_storm,
    max_workers,
    False
)

# Replace only those rows in the main DataFrame
sinclim_unique_wgust_min_claims.update(claims_to_update)

# Drop rows where wgust_max is still missing
sinclim_unique_wgust_min_claims = sinclim_unique_wgust_min_claims.dropna(subset=['wgust_max'])

path_sinclim_storm_wgust_min_claims = (
    PATH_TEMP
    + "final_per_year/" 
    + "r-varying/"
    + f"{save_name}_d-{d_before}_d+{d_after}_unique-wgust_min{nb_min_claims}_{track_source}_{input_year}-{end_year}_r-varying.csv"
)
sinclim_unique_wgust_min_claims.to_csv(path_sinclim_storm_wgust_min_claims, encoding='utf-8', index=False)