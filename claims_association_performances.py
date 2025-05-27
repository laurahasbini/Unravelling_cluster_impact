## This script run the association method and compute the performances for 
## The input period should be filled as
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
from fct.paths import *

# This funcion must be called as a loop over the years of interest (1998-2023) with the beginning year of the season as argument
# For example, is the year 2010 is written as input, the method will compute the performances of the association for the season 2010-2011
if __name__ == "__main__":
    mp.set_start_method('fork', force=True)
    client = setup_dask()
    #global input_year
    input_year = int(sys.argv[1])

    ############ Open the needed data and perform the association 
    sinclim_version = 'v2.1'
    period = '1979-2024WIN'
    r = 1300

    # Varying parameters 
    method = "wgust"
    Xb_range = [3, 2, 1, 0]
    Xa_range = [5, 4, 3, 2]
    claims_range = np.arange(30, 110, 10)
    local_height = 10

    performances = pd.DataFrame(columns = ['Xb', 'Xa', 'method', 'min_claims', 'frequency_diff', 'max_days_diff', "percentage_loss", "percentage_claims"])

    if sinclim_version == 'v1' :
        sinclim = pq.read_table(PATH_GENERALI+'sinclim_anom.parquet')
    elif sinclim_version =='v2.1' : 
        sinclim = pq.read_table(PATH_GENERALI+'sinclim_v2.1_anom.parquet')

    sinclim_pd      =  sinclim.to_pandas(date_as_object = True, safe = False)
    sinclim_pd_sort = sinclim_pd.sort_values('dat_sin')

    preprocess_start = time.time()
    type_claim = 'tempete' #'inondation'
    if type_claim == "tempete" :
        save_name = "sinclim_"+sinclim_version+"_storm"
    elif type_claim == "inondation" :
        save_name = "sinclim_"+sinclim_version+"_flood"

    ### Filter the sinclim data
    sinclim_sort_storm = fct_link_storm_claim.claims_preprocess(sinclim_pd_sort, type_claim)
    preprocess_end = time.time()
#     print(f"Preprocessing claims data took {preprocess_end - preprocess_start:.2f} seconds")

    ### OPEN STORM TRAJECTORIES --> ONLY FOR 2018 at a start 
    track_source = 'priestley_ALL'
    df_info_storm_priestley = pd.read_csv(PATH_TRACKS+"tracks_ALL_24h_"+period+"_info.csv", encoding='utf-8')
    df_info_storm_priestley['storm_landing_date'] = pd.to_datetime(df_info_storm_priestley['storm_landing_date'])
    df_info_storm_priestley = df_info_storm_priestley.sort_values('storm_landing_date')
    df_storm_priestley = pd.read_csv(PATH_TRACKS+"tracks_ALL_24h_"+period+".csv", encoding='utf-8')

    # Trajectories between the input period 
    df_info_storm_priestley = df_info_storm_priestley.loc[df_info_storm_priestley.storm_landing_date < datetime.datetime(year=input_year+1, 
                                                                                                                         month=4, day=1, hour=0
                                                                                                                        )]
#     df_storm_priestley = df_storm_priestley.loc[df_storm_priestley.storm_id.isin(df_info_storm_priestley.storm_id.unique())]
    df_info_storm_priestley = df_info_storm_priestley.loc[df_info_storm_priestley.storm_landing_date > datetime.datetime(year=input_year, 
                                                                                                                         month=10, day=1, hour=0
                                                                                                                        )]
    df_storm_priestley = df_storm_priestley.loc[df_storm_priestley.storm_id.isin(df_info_storm_priestley.storm_id.unique())]
    
    for d_after in Xa_range : 
        for d_before in Xb_range : 
            ###LINK STORM AND DATA 
            delta_t_after = datetime.timedelta(hours=d_after * 24)
            delta_t_before =  datetime.timedelta(hours=-d_before * 24)
        
            # Select raw sinclim data 
            sinclim_raw = sinclim_sort_storm.loc[sinclim_sort_storm.dat_sin < datetime.datetime(year=input_year+1, month=4, day=1, hour=0)+delta_t_after].copy()
            sinclim_raw = sinclim_raw.loc[sinclim_raw.dat_sin > datetime.datetime(year=input_year, month=10, day=1, hour=0)-delta_t_before]
            
            #For the largest windows make the association from scratch
            if (d_after == 5) & (d_before==3) : 
                sinclim_storm = fct_link_storm_claim.assoc_storm_candidates(sinclim_raw, d_before, d_after, df_storm_priestley, df_info_storm_priestley)

                ## ADD WGUST
                wgust_start = time.time()
                combined_dataset_storm = fct_link_storm_claim.import_wgust_footprint(sinclim_storm, r)
                sinclim_storm = fct_link_storm_claim.add_wgust_new(sinclim_storm, combined_dataset_storm)
                sinclim_storm = sinclim_storm.compute()
                sinclim_full_window = sinclim_storm.copy()
                wgust_end = time.time()
            #For other windows, re-use the results of the association 
            else : 
                sinclim_storm = fct_link_storm_claim.assoc_storm_candidates(sinclim_full_window, d_before, d_after, df_storm_priestley, df_info_storm_priestley, update=True)

            #### Link to Higest windgust at location 
            # Make a copy and drop NA values
            sinclim_copy2 = sinclim_storm.copy()

            # Identify locations with multiple storms
            grouped = sinclim_copy2.groupby('cod_sin')['storm_id'].nunique().reset_index(name='unique_storm_ids')
            multi_storm_cod_sins = grouped[grouped['unique_storm_ids'] > 1]['cod_sin']

            # For locations with multiple storms, keep only the row with maximum wgust_max
            if not multi_storm_cod_sins.empty:
                # Get indices of max rows for multi-storm locations
                idx_to_keep = (
                    sinclim_copy2[sinclim_copy2['cod_sin'].isin(multi_storm_cod_sins)]
                    .groupby('cod_sin')['wgust_max']
                    .idxmax()
                )

                # Directly filter using these indices
                sinclim_copy2 = sinclim_copy2[
                    ~sinclim_copy2['cod_sin'].isin(multi_storm_cod_sins) | 
                    sinclim_copy2.index.isin(idx_to_keep)
                ]

            ### Apply gathering of number of claims
            nb_min_claims_start = 10
            sinclim_perf = sinclim_copy2.copy()#.compute()
            for nb_min_claims in claims_range : 
                sinclim_perf = fct_link_storm_claim.gather_claims_storm_iteration(sinclim_perf, df_info_storm_priestley, nb_min_claims, 
                                                                                   nb_min_claims_start, 10)
    
                nb_min_claims_start = nb_min_claims
                percentage_loss = sinclim_perf.num_chg_brut.sum() / sinclim_raw.num_chg_brut.sum()
                percentage_claims = len(sinclim_perf)/len(sinclim_raw)
            
                #### Compute the performances and save 
                frequency_diff, minimal_dates_diffs = fct_link_storm_claim.performance(sinclim_raw, sinclim_perf, df_info_storm_priestley, local_height)
                new_row = pd.DataFrame({"Xb"                : [d_before], 
                                        "Xa"                : [d_after], 
                                        "method"            : ["wgust"], 
                                        "min_claims"        : [nb_min_claims], 
                                        "frequency_diff"    : [frequency_diff], 
                                        "max_days_diff"     : [np.max(minimal_dates_diffs)], 
                                        "percentage_loss"   : [percentage_loss], 
                                        "percentage_claims" : [percentage_claims]})
                performances = pd.concat([performances, new_row], ignore_index=True)
                performances.to_csv(PATH_GENERALI+"performances_local-height-"+str(local_height)+"_r"+str(r)+"_"+str(input_year)+"-"+str(input_year+1)+".csv" , encoding='utf-8', index=False)

    performances.to_csv(PATH_GENERALI+"performances_local-height-"+str(local_height)+"_r"+str(r)+"_"+str(input_year)+"-"+str(input_year+1)+".csv" , encoding='utf-8', index=False)
