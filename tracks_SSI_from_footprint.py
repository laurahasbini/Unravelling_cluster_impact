## The following script, compute the SSI from the footprint 
import math
import os
import pandas as pd
import numpy as np
import xarray as xr 
import datetime 
import csv
from scipy.stats import kde
from haversine import haversine
import netCDF4
import xskillscore as xs
import geopandas as gpd
from scipy import stats
import rioxarray 

#Masking array
from rechunker import rechunk
from rasterio import features
from affine import Affine

#External functions 
from fct.paths import *
import fct.storm_eu as storm_eu

#Load data
quantile = ['90', '95', '98']

exposure = xr.open_mfdataset(PATH_GENERALI+"exposure.nc", 
                                chunks={'latitude':'auto', 'longitude':'auto'})
exposure_bool = True 
exposure_var = "nb_policies_norm"

for q in quantile : 
    wind_th = xr.open_mfdataset(PATH_WGUST_QUANTILE+"windgust_percentile-q"+q+"_1950_2022_winter.nc", 
                                chunks={'time':-1, 'latitude':'auto', 'longitude':'auto'})
    wind_th['fg10'] = xr.where(wind_th['fg10'] < 9, 9, wind_th['fg10'])

    FRA_geo_shp = gpd.read_file(os.path.join(PATH_SHP_COUNTRIES, "world-administrative-boundaries_FRA_level0.shp"))

    ### OPEN STORM TRAJECTORIES --> ONLY FOR 2018 at a start 
    track_source = 'priestley_ALL'
    period = "1979-2024WIN"
    df_info_storm = pd.read_csv(PATH_TRACKS+"tracks_FR_ALL_24h_"+period+"_info.csv", encoding='utf-8')
    df_info_storm['storm_landing_date'] = pd.to_datetime(df_info_storm['storm_landing_date'])
    df_info_storm = df_info_storm.loc[df_info_storm.storm_landing_date >= datetime.datetime(year=1997, month=12, day=1)]

    ### COMPUTE SSI
    r=1300
    if exposure_bool :
        df_info_storm_SSI = storm_eu.SSI_from_footprint(df_info_storm, 'SSI_FRA_wgust_q'+q, 
                                                                  PATH_FOOTPRINTS, 'max_wind_gust', r, 
                                                                  wind_th.fg10, 
                                                                  True, FRA_geo_shp, 
                                                                  True, PATH_TRACKS, "tracks_FR_ALL_24h_"+period+"_info_SSI-"+exposure_var+"-wgust-q"+q+"_r"+str(r), 
                                                                  True, exposure, exposure_var)
        df_info_storm_SSI.to_csv(PATH_TRACKS+"tracks_FR_ALL_24h_"+period+"_info_SSI-"+exposure_var+"-wgust-q"+q+"_r"+str(r)+".csv" , encoding='utf-8', index=False)
    else : 
        df_info_storm_SSI = storm_eu.SSI_from_footprint(df_info_storm, 'SSI_FRA_wgust_q'+q, 
                                                                  PATH_FOOTPRINTS, 'max_wind_gust', r, 
                                                                  wind_th.fg10, 
                                                                  True, FRA_geo_shp, 
                                                                  True, PATH_TRACKS, "tracks_FR_ALL_24h_"+period+"_info_SSI-wgust-q"+q+"_r"+str(r), 
                                                                  False)
        df_info_storm_SSI.to_csv(PATH_TRACKS+"tracks_FR_ALL_24h_"+period+"_info_SSI-wgust-q"+q+"_r"+str(r)+".csv" , encoding='utf-8', index=False)