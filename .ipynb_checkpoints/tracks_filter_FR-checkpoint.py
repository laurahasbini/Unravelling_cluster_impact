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
import fct.storm_eu as storm_eu
from fct.paths import *

########### IMPORT STORMS 
cat_type                                       = ['ALL_24h']
period                                         = "1979-2024WIN"

#Tracks lasting more than 24h
df_info_storm                       = pd.read_csv(PATH_TRACKS+"tracks_ALL_24h_"+period+"_info.csv", encoding='utf-8')
df_info_storm['storm_landing_date'] = pd.to_datetime(df_info_storm['storm_landing_date'])
df_storm                            = pd.read_csv(PATH_TRACKS+"tracks_ALL_24h_"+period+".csv", encoding='utf-8')

# Import de the Europe shapefile 
FRA_geo_shp = gpd.read_file(os.path.join(PATH_SHP_COUNTRIES, "world-administrative-boundaries_FRA_level0.shp"))

#Filter the ones having a now null intersection with FR
df_info_storm_FR = storm_eu.storm_over_area(df_info_storm, df_storm, r=1300, is_mask=True, mask=FRA_geo_shp)
df_info_storm_FR.to_csv(PATH_TRACKS+"tracks_FR_ALL_24h_"+period+"_info.csv" , encoding='utf-8', index=False)

df_storm_FR = df_storm.loc[df_storm.storm_id.isin(df_info_storm_FR.storm_id)]
df_storm_FR.to_csv(PATH_TRACKS+"tracks_FR_ALL_24h_"+period+".csv" , encoding='utf-8', index=False)
