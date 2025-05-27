## This script run the association method
import multiprocessing as mp
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

def setup_dask():
    """Initialize Dask cluster properly."""
    global client, cluster
    cluster = LocalCluster()
    client = Client(cluster)
    return client

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

# This funcion must be called as a loop over the years of interest (1998-2023) with the beginning year of the season as argument
# For example, is the year 2010 is written as input, the method will association the claims for the season 2010-2011
if __name__ == "__main__":
    mp.set_start_method('fork', force=True)
    client = setup_dask()
    input_year = int(sys.argv[1])
    
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
    df_info_storm_priestley = pd.read_csv(PATH_TRACKS+"tracks_ALL_24h_"+period+"_info.csv", encoding='utf-8')
    df_info_storm_priestley['storm_landing_date'] = pd.to_datetime(df_info_storm_priestley['storm_landing_date'])
    df_info_storm_priestley = df_info_storm_priestley.sort_values('storm_landing_date')
    df_storm_priestley = pd.read_csv(PATH_TRACKS+"tracks_ALL_24h_"+period+".csv", encoding='utf-8')

    # Trajectories between the input period 
    
    df_info_storm_priestley = df_info_storm_priestley.loc[df_info_storm_priestley.storm_landing_date < datetime.datetime(year=input_year+1, 
                                                                                                                         month=4, day=1, hour=0
                                                                                                                        )]
    df_info_storm_priestley = df_info_storm_priestley.loc[df_info_storm_priestley.storm_landing_date >= datetime.datetime(year=input_year, 
                                                                                                                     month=10, day=1, hour=0
                                                                                                                    )]
    df_storm_priestley = df_storm_priestley.loc[df_storm_priestley.storm_id.isin(df_info_storm_priestley.storm_id.unique())]

    ###LINK STORM AND DATA 
    delta_t_after = datetime.timedelta(hours=d_after * 24)
    delta_t_before =  datetime.timedelta(hours=-d_before * 24)

    # Select raw sinclim data 
    sinclim_raw = sinclim_sort_storm.loc[sinclim_sort_storm.dat_sin < datetime.datetime(year=input_year+1, month=4, day=1, hour=0)+delta_t_after].copy()
    sinclim_raw = sinclim_raw.loc[sinclim_raw.dat_sin >= datetime.datetime(year=input_year, month=10, day=1, hour=0)-delta_t_before]
    
#     sinclim_raw = sinclim_sort_storm.loc[sinclim_sort_storm.dat_sin < datetime.datetime(year=2024+1, month=4, day=1, hour=0)+delta_t_after].copy()
#     sinclim_raw = sinclim_raw.loc[sinclim_raw.dat_sin > datetime.datetime(year=1998, month=1, day=1, hour=0)-delta_t_before]
    
    #For the largest windows make the association from scratch
    sinclim_storm = fct_link_storm_claim.assoc_storm_candidates(sinclim_raw, d_before, d_after, df_storm_priestley, df_info_storm_priestley)
    sinclim_storm.to_csv(PATH_GENERALI+"final/"+save_name+"_d-"+str(d_before)+"_d+"+str(d_after)+"_"+track_source+"_"+str(input_year)+"-"+str(input_year+1)+".csv" , encoding='utf-8', index=False)
    
    ## ADD WGUST
    combined_dataset_storm = fct_link_storm_claim.import_wgust_footprint(sinclim_storm, r)
    sinclim_storm = fct_link_storm_claim.add_wgust_new(sinclim_storm, combined_dataset_storm)
    sinclim_storm = sinclim_storm.compute()
    sinclim_storm.to_csv(PATH_GENERALI+"final/"+save_name+"_d-"+str(d_before)+"_d+"+str(d_after)+"_"+track_source+"_"+str(input_year)+"-"+str(input_year+1)+"_r"+str(r)+".csv" , encoding='utf-8', index=False)

    #### Link to Higest windgust at location 

    # Make a copy and drop NA values
    sinclim_copy_cp = sinclim_storm.copy()#.dropna(subset=['wgust_max'])

    # Identify locations with multiple storms
    grouped = sinclim_copy_cp.groupby('cod_sin')['storm_id'].nunique().reset_index(name='unique_storm_ids')
    multi_storm_cod_sins = grouped[grouped['unique_storm_ids'] > 1]['cod_sin']

    # For locations with multiple storms, keep only the row with maximum wgust_max
    if not multi_storm_cod_sins.empty:
        # Get indices of max rows for multi-storm locations
        idx_to_keep = (
            sinclim_copy_cp[sinclim_copy_cp['cod_sin'].isin(multi_storm_cod_sins)]
            .groupby('cod_sin')['wgust_max']
            .idxmax()
        )

        # Directly filter using these indices
        sinclim_copy_cp = sinclim_copy_cp[
            ~sinclim_copy_cp['cod_sin'].isin(multi_storm_cod_sins) | 
            sinclim_copy_cp.index.isin(idx_to_keep)
        ]

    ### Apply gathering of number of claims
    nb_min_claims_start = 10

    sinclim_copy_cp = fct_link_storm_claim.gather_claims_storm_iteration(sinclim_copy_cp, df_info_storm_priestley, nb_min_claims, 
                                                                       nb_min_claims_start, 10)

    #Correct the final wgust 
    sinclim_copy_cp = sinclim_copy_cp.drop('wgust_max', axis=1)
    sinclim_copy_cp = fct_link_storm_claim.add_wgust_new(sinclim_copy_cp, combined_dataset_storm)
    sinclim_copy_cp = sinclim_copy_cp.dropna(subset=['wgust_max'])
    sinclim_copy_cp = sinclim_copy_cp.compute()

    sinclim_copy_cp.to_csv(PATH_GENERALI+"final/"+save_name+"_d-"+str(d_before)+"_d+"+str(d_after)+"_unique-wgust_min"+str(50)+"_"+track_source+"_"+str(input_year)+"-"+str(input_year+1)+"_r"+str(r)+".csv" , encoding='utf-8', index=False)