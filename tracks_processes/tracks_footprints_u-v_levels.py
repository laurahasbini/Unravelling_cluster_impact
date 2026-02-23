import math
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr 
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import datetime 
import csv
import seaborn as sns
from scipy.stats import kde
import plotly.express as px
from haversine import haversine
from cartopy.geodesic import Geodesic
import shapely.geometry as sgeom
import netCDF4
import xskillscore as xs
import geopandas as gpd
from scipy import stats
import rioxarray 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import dask
dask.config.set(**{'array.slicing.split_large_chunks': False})

proj = ccrs.PlateCarree()

#Masking array
from rechunker import rechunk
from rasterio import features
from affine import Affine
from shapely.geometry import Point

#External functions 
from fct.paths import *
import fct.storm_eu as storm_eu
import fct.preprocess_sinclim as preprocess_sinclim

###Import data 
###Wgust 
r=1300
period = "1979-2024WIN"

###Tracks
df_info_storm                       = pd.read_csv(PATH_TRACKS+"tracks_ALL_24h_"+period+"_info.csv", encoding='utf-8')
df_info_storm['storm_landing_date'] = pd.to_datetime(df_info_storm['storm_landing_date'])
df_info_storm                       = df_info_storm.sort_values('storm_landing_date')
df_storm                            = pd.read_csv(PATH_TRACKS+"tracks_ALL_24h_"+period+".csv", encoding='utf-8')

# ## Subselect storms 
# df_info_storm                       = df_info_storm.loc[df_info_storm.storm_landing_date < datetime.datetime(year=1997, month=1, day=1, hour=0)]
# df_storm                            = df_storm.loc[df_storm.storm_id.isin(df_info_storm.storm_id.unique())]

## Subselect storms which were identified with impact 
window        = 'd-3_d+3'
min_claim     = 50
method        = 'wgust'
r             = 1300
sinclim = preprocess_sinclim.open_sinclim_associated(PATH_GENERALI, window, min_claim, method, period, r)
stormi_impact = np.unique(sinclim.storm_id)
df_info_storm                       = df_info_storm.loc[df_info_storm.storm_id.isin(stormi_impact)]
df_storm                            = df_storm.loc[df_storm.storm_id.isin(stormi_impact)]

### IF SAVE IN SHP 
# nc_file, gdf = storm_eu.footprint_shp(df_storm, 
#                        PATH_WGUST, 
#                        'windgust_10m_hourly', 
#                        'fg10',
#                        'y',
#                        'max',
#                        PATH_FOOTPRINTS,
#                        is_mask=True,
#                        mask=FRA_geo_shp, r=r)

### IF SAVE IN NC 
for plev in [900, 250] : 
    for var in ['u', 'v'] : 
        PATH_WIND = f"/home/estimr3/bdd/ERA5/North_Atlantic_22p5N_70N_80W_50E/6hourly/{var}{plev}/"
        print(PATH_WIND)
        PATH_SAVE_FOOTPRINT = f"/home/estimr3/lhasbini/data_storm/priestley/footprint/NH_ALL_IMPACT/{var}{plev}/"
        nc_file = storm_eu.footprint_nc(df_storm, 
                                       df_info_storm,
                                       PATH_WIND, 
                                       f"{var}wind_{plev}hPa", 
                                       var,
                                       'y',
                                       'max',
                                        True, 
                                       PATH_SAVE_FOOTPRINT,
                                       is_mask=False,
                                       mask=None, 
                                       r=r)