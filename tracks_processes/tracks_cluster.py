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
df_info_storm                                  = {}
df_storm                                       = {}
cat_type                                       = ['ALL_24h']
period                                         = "1979-2024WIN"

#ALL the tracks
df_info_storm['ALL']                           = pd.read_csv(PATH_TRACKS+"tracks_ALL_"+period+"_info.csv", encoding='utf-8')
df_info_storm['ALL']['storm_landing_date']     = pd.to_datetime(df_info_storm['ALL']['storm_landing_date'])
df_storm['ALL']                                = pd.read_csv(PATH_TRACKS+"tracks_ALL_"+period+".csv", encoding='utf-8')

#Tracks lasting more than 24h
df_info_storm['ALL_24h']                       = pd.read_csv(PATH_TRACKS+"tracks_FR_ALL_24h_"+period+"_info.csv", encoding='utf-8')
df_info_storm['ALL_24h']['storm_landing_date'] = pd.to_datetime(df_info_storm['ALL_24h']['storm_landing_date'])
df_storm['ALL_24h']                            = pd.read_csv(PATH_TRACKS+"tracks_FR_ALL_24h_"+period+".csv", encoding='utf-8')

# Import de the Europe shapefile 
shp_path = "/home/estimr3/lhasbini/shp_files/"
FRA_geo_shp = gpd.read_file(os.path.join(PATH_SHP_COUNTRIES, "world-administrative-boundaries_FRA_level0.shp"))

########## CLUSTER 
cluster_window = [96, 72]
r = 700
region = 'FRA'

for nb_hours in cluster_window : 
    df_cluster = {}
    df_info_cluster = {}

    for cat in cat_type :         
        #### RUN FOR STORM IN MULTIPLE CLUSTERS 
        df_info_cluster = storm_eu_cluster.assign_clusters_allow_multiple(df_info_storm[cat], df_storm[cat], r=r, nb_hours_diff=nb_hours, is_mask=True, mask=FRA_geo_shp)
        df_info_mult_cluster = storm_eu_cluster.filter_clusters_explode(df_info_cluster)
        df_mult_cluster = df_storm[cat].loc[df_storm[cat].storm_id.isin(df_info_mult_cluster.storm_id.unique())]
        
        df_mult_cluster.to_csv(PATH_TRACKS+"tracks_"+cat+"_clust-mult-2storms-"+str(nb_hours)+"h_r"+str(r)+"_FRA_"+period+".csv" , encoding='utf-8', index=False)
        df_info_mult_cluster.to_csv(PATH_TRACKS+"tracks_"+cat+"_clust-mult-2storms-"+str(nb_hours)+"h_r"+str(r)+"_FRA_"+period+"_info.csv" , encoding='utf-8', index=False)       