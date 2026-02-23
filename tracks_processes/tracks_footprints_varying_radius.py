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

###Import data 
###Wgust 
r=1300
period = "1979-2024WIN"

###Tracks
df_info_storm                       = pd.read_csv(PATH_TRACKS+"tracks_ALL_24h_"+period+"_info.csv", encoding='utf-8')
df_info_storm['storm_landing_date'] = pd.to_datetime(df_info_storm['storm_landing_date'])
df_info_storm                       = df_info_storm.sort_values('storm_landing_date')
df_storm                            = pd.read_csv(PATH_TRACKS+"tracks_ALL_24h_"+period+".csv", encoding='utf-8')

## Subselect storms 
df_info_storm                       = df_info_storm.loc[df_info_storm.storm_landing_date >= datetime.datetime(year=1997, month=1, day=1, hour=0)]
df_storm                            = df_storm.loc[df_storm.storm_id.isin(df_info_storm.storm_id.unique())]

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
nc_file = storm_eu.footprint_nc_varying_radius(df_storm, 
                                       df_info_storm,
                                       PATH_WGUST, 
                                       'windgust_10m_hourly', 
                                       'fg10',
                                       'y',
                                       'max',
                                        True, 
                                       PATH_FOOTPRINTS_VARYING_RADIUS,
                                       is_mask=False,
                                       mask=None)