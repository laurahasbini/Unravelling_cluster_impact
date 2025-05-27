########## IMPORT 
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
from haversine import haversine
import shapely.geometry as sgeom
import netCDF4
import xskillscore as xs
import geopandas as gpd
from scipy import stats
import rioxarray
import re

#Masking array
from rechunker import rechunk
from rasterio import features
from affine import Affine
from shapely.geometry import Point
from shapely.geometry import Polygon

#External functions 
import fct.storm_eu as storm_eu
from fct.paths import * 

######### OPEN PRIMAVERA DATA 
periods         = ["DJF", "SON", "MAM"]
years_by_period = {"DJF" : np.arange(1979, 2024), 
                   "SON" : np.arange(1979, 2024), 
                   "MAM" : np.arange(1979, 2024)
                  }

filt          = "ALL"

if filt == "ALL" :
    filtered_name = filt
elif filt == "FILT" : 
    filtered_name = "FILTERED"
    
for period in periods : 
    years = years_by_period[period]
    for year in years :
        PATH_TRACKS_RAW = "/home/estimr3/lhasbini/data_storm/priestley/"+period+"/NH_"+filt+"/"

        if   period == "DJF" : 
            data_init  = datetime.datetime(year=year, month=12, day=1)
            file_path      = PATH_TRACKS_RAW+"ERA5_"+str(year)+str(year+1)+"_TRACKS_"+filtered_name+"_pos.addmslp"
        elif period == "SON" : 
            data_init  = datetime.datetime(year=year, month=9, day=1)
            file_path      = PATH_TRACKS_RAW+"ERA5_"+str(year)+"_SON_TRACKS_"+filtered_name+"_pos.addmslp"
        elif period == "MAM" : 
            data_init  = datetime.datetime(year=year, month=3, day=1)
            file_path      = PATH_TRACKS_RAW+"ERA5_"+str(year)+"_MAM_TRACKS_"+filtered_name+"_pos.addmslp"

        # Variables to store data
        tracks     = []
        track_id = None

        # Regular expressions to match track and point lines
        track_id_pattern = re.compile(r"TRACK_ID\s+(\d+)")
        point_pattern = re.compile(r"(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.eE+-]+)\s*&\s*([\d.eE+-]+)\s*&\s*([\d.eE+-]+)\s*&\s*([\d.eE+-]+)\s*&")

        # Open and read the file
        with open(file_path, 'r') as file:
            for line in file:
                track_match = track_id_pattern.search(line)
                if track_match:
                    track_id = int(track_match.group(1))
                else:
                    point_match = point_pattern.search(line)
                    if point_match and track_id is not None:
                        # Extract data points
                        point_num = int(point_match.group(1))
                        lon = float(point_match.group(2))
                        lat = float(point_match.group(3))
                        vo_max = float(point_match.group(4))
                        lon_mslp = float(point_match.group(5))
                        lat_mslp = float(point_match.group(6))
                        slp = float(point_match.group(7))

                        # Append to list
                        tracks.append([track_id, point_num, lon, lat, vo_max, lon_mslp, lat_mslp, slp])

        # Convert the list of data to a DataFrame
        df = pd.DataFrame(tracks, columns=[
            'track_id', 'time', 
            'lon', 'lat', 'vo_max', 
            'lon_mslp', 'lat_mslp', 'slp'
        ])
        df = df.iloc[:-1]
        df = df.dropna()

        ####### CONVERT THE DATA TO TE FORMAT 

        # Remove the columns not present in TE
        # columns_to_remove = ['latitude_vo_max', 'longitude_vo_max', 'vo_max', 'wind925_max_3deg']
        # df = df.drop(columns=columns_to_remove)

        #Convert the time to the same format 
        df['time']  = [data_init + datetime.timedelta(hours=6*(delta-1)) for delta in df['time']]
        df['year']  = df['time'].dt.year
        df['month'] = df['time'].dt.month
        df['day']   = df['time'].dt.day
        df['hour']  = df['time'].dt.hour
        df          = df.drop(columns=['time'])

        ####Convert longitude
        ## Change the latitude/longitude used 
        df = df.drop(['lon', 'lat'], axis=1)
        df = df.rename(columns={'lon_mslp': 'lon', 'lat_mslp': 'lat'})
        
        longitude_columns = ['lon', 'lat']
        for col in longitude_columns:
            df[col] = df[col].astype(float)  # Ensure the columns are in float format
            df[col] = df[col].apply(lambda x: x if x <= 180 else x - 360)

        ####Convert pressure to Pa
        df['slp'] = pd.to_numeric(df['slp'], errors='coerce')
        df['slp'] = df['slp']*100

        ####Remove the rows without msl lat/lon 
        df = df[(df['lat'] != 1e25) & (df['lon'] != 1e25)]

        ##### Filter on the North Atlantic tracks 
        minlon, maxlon = -30.0, 0
        minlat, maxlat = 30.0, 65.0
        tracks_in_basin = df[
            (df['lon'] >= minlon) & (df['lon'] <= maxlon) & 
            (df['lat'] >= minlat) & (df['lat'] <= maxlat)
        ]['track_id'].unique()
        df = df[df['track_id'].isin(tracks_in_basin)]

        ####### CREATE STORM_INFO FILE 
        df_storm_info = pd.DataFrame(columns = ['track_id', 'min_slp', 'storm_landing_date'])
        unique_track_loop = [i for i in np.unique(df.track_id)] # if i>max(df_storm_output.track_id)

        for i in unique_track_loop :
            cycl=df[df.track_id==i]        

            #Compute storm date 
            cycl_shift = cycl.copy()
            cycl_shift.lon = np.abs(cycl_shift.lon+7.5)
            cycl_lon_eu = cycl_shift.loc[cycl_shift.lon == min(cycl_shift.lon)]
            if(len(cycl_lon_eu.index)>1) :
                date_storm = datetime.datetime(year=int(cycl_lon_eu.year.iloc[0]),
                                   month=int(cycl_lon_eu.month.iloc[0]),
                                   day=int(cycl_lon_eu.day.iloc[0]),
                                   hour=int(cycl_lon_eu.hour.iloc[0]))
            else : 
                date_storm = datetime.datetime(year=int(cycl_lon_eu.year),
                                               month=int(cycl_lon_eu.month),
                                               day=int(cycl_lon_eu.day),
                                               hour=int(cycl_lon_eu.hour))

            df_list_info = pd.DataFrame({'track_id' : cycl['track_id'].iloc[0],
                            'min_slp' : min(cycl.slp),
                            'storm_landing_date' : date_storm}, index = [len(df_storm_info)])
            df_storm_info = pd.concat([df_storm_info, df_list_info])

        ####### CHANGE THE IDENTIFIER 
        df['track_id_origin']            = df['track_id']
        df_storm_info['track_id_origin'] = df_storm_info['track_id']
        df, df_storm_info = storm_eu.change_identifier(df, df_storm_info)

        # Save info DataFrame
        df.to_csv(PATH_TRACKS+"season/"+"tracks_mslp_"+filt+"_"+period+"_"+str(year)+str(year+1)+".csv" , encoding='utf-8', index=False)  
        df_storm_info.to_csv(PATH_TRACKS+"season/"+"tracks_mslp_"+filt+"_"+period+"_"+str(year)+str(year+1)+"_info.csv" , encoding='utf-8', index=False)