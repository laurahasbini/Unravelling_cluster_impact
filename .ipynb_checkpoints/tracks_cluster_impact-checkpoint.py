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
from scipy.stats import gaussian_kde
from haversine import haversine
import netCDF4
import xskillscore as xs
import geopandas as gpd
from scipy import stats
import rioxarray 

#External functions 
import fct.storm_eu_cluster as storm_eu_cluster
from fct.paths import *

########### IMPORT STORMS
period = "1979-2024WIN"

df_info_storm                                  = {}
df_storm                                       = {}
cat_type                                       = ['ALL_24h']

#Tracks lasting more than 24h
df_info_storm['ALL_24h_original']                       = pd.read_csv(PATH_TRACKS+"tracks_ALL_24h_"+period+"_info.csv", encoding='utf-8')
df_info_storm['ALL_24h_original']['storm_landing_date'] = pd.to_datetime(df_info_storm['ALL_24h_original']['storm_landing_date'])
df_storm['ALL_24h_original']                            = pd.read_csv(PATH_TRACKS+"tracks_ALL_24h_"+period+".csv", encoding='utf-8')

# Import de the Europe shapefile 
FRA_geo_shp = gpd.read_file(os.path.join(PATH_SHP_COUNTRIES, "world-administrative-boundaries_FRA_level0.shp"))

########## IMPACT DATA 
window        = "d-3_d+3"
min_claims_range = [50]
method        = 'wgust'

########## CLUSTER 
cluster_window = [72, 96, 120]
radius = [700]
r_sinclim = 1300

for min_claim in min_claims_range : 
    sinclim       = pd.read_csv(PATH_GENERALI+"sinclim_v2.1_storm_"+window+"_unique-"+method+"_min"+str(min_claim)+"_priestley_ALL_"+period+"_r"+str(r_sinclim)+".csv", low_memory=False)
    stromi_impact = sinclim.storm_id.unique()
    
    df_info_storm['ALL_24h'] = df_info_storm['ALL_24h_original'].loc[df_info_storm['ALL_24h_original'].storm_id.isin(stromi_impact)]
    df_storm['ALL_24h']      = df_storm['ALL_24h_original'].loc[df_storm['ALL_24h_original'].storm_id.isin(stromi_impact)]
    for nb_hours in cluster_window : 
        for r in radius : 
            df_cluster = {}
            df_info_cluster = {}

            for cat in cat_type : 

                #### RUN FOR STORM IN MULTIPLE CLUSTERS 
                df_info_cluster = storm_eu_cluster.assign_clusters_allow_multiple(df_info_storm[cat], df_storm[cat], r=r, nb_hours_diff=nb_hours)
                df_info_mult_cluster = storm_eu_cluster.filter_clusters_explode(df_info_cluster)
                df_mult_cluster = df_storm[cat].loc[df_storm[cat].storm_id.isin(df_info_mult_cluster.storm_id.unique())]

                df_mult_cluster.to_csv(PATH_TRACKS+"tracks_"+cat+"_impact_"+window+"unique-"+method+"_min"+str(min_claim)+"_clust-mult-2storms-"+str(nb_hours)+"h_r"+str(r)+"_"+period+".csv" , encoding='utf-8', index=False)
                df_info_mult_cluster.to_csv(PATH_TRACKS+"tracks_"+cat+"_impact_"+window+"unique-"+method+"_min"+str(min_claim)+"_clust-mult-2storms-"+str(nb_hours)+"h_r"+str(r)+"_"+period+"_info.csv" , encoding='utf-8', index=False)
        

