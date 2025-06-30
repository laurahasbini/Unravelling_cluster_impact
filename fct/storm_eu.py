import numpy as np
import pandas as pd
import xarray as xr
import os
import datetime
import rioxarray 

from rechunker import rechunk
from rasterio import features
from affine import Affine

import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.ops import unary_union

from pyproj import Transformer
from shapely.ops import transform

import geopandas as gpd
import shapely.geometry as sgeom
from shapely.geometry import Point, box
import matplotlib.pyplot as plt

from fct.paths import *

# Shit the longitude to have -180/180
def shift_longitude(ds):
    # Shift longitude from 0-360 to -180-180
    ds = ds.copy()  # To avoid modifying the original dataset
    ds['longitude'] = np.where(ds['longitude'] > 180, ds['longitude'] - 360, ds['longitude'])

    # Sort by longitude in case the order is affected
    ds = ds.sortby('longitude')
    
    return ds

# Function to project and calculate area
def calculate_area(geometry):
    projected = transform(transformer.transform, geometry)
    return projected.area

def storm_over_area(df_track_info, df_tracks, r=700, is_mask=False, mask=None):
    # Convert landing dates to datetime and create geometries
    df_track_info['storm_landing_date'] = pd.to_datetime(df_track_info['storm_landing_date'])
    gdf_tracks = gpd.GeoDataFrame(
        df_tracks,
        geometry=gpd.points_from_xy(df_tracks.lon, df_tracks.lat),
        crs="EPSG:4326"
    )
    gdf_tracks['buffer'] = gdf_tracks.to_crs(epsg=3395).geometry.buffer(r * 1000).to_crs(epsg=4326)
    spatial_index = gdf_tracks.sindex
    
    storm_id_save = []

    for _, storm_row in df_track_info.iterrows():
        storm_id = storm_row['storm_id']
        storm_date = storm_row['storm_landing_date']
        storm_buffer = gdf_tracks.loc[gdf_tracks['storm_id'] == storm_id, 'buffer'].values
        union_buffer = unary_union(storm_buffer)
        combined_union = union_buffer
        
        #Apply an extra filter over the region
        if is_mask : 
            for geom in mask.geometry :
                union_buffer = union_buffer.intersection(geom)
                if union_buffer.is_empty:
                    break
                
        if not union_buffer.is_empty:
            storm_id_save.append(storm_id)
    
    return df_track_info.loc[df_track_info.storm_id.isin(storm_id_save)]

#Change of identifyer

def change_identifier(all_tracks, df_storm_info) :
    """
    INPUT 
        all_tracks : DataFrame of storm trakcs in the TempestExtreme output format 
        df_storm_info : DataFrame with one row per European storm. Constains general informations about the storm (surface impacted, minimal slp ....) 
    OUTPUT 
        all_tracks_new : Same DataFrame as above but track_id is removed and replaced by the storm landing date
    """
    #Create the new DataFrame
    all_tracks_new = pd.DataFrame(columns= all_tracks.columns)
    all_tracks_new = all_tracks_new.drop('track_id', axis=1)
    all_tracks_new.insert(0, 'storm_id', None)
    df_storm_info_new = pd.DataFrame(columns= df_storm_info.columns)
    df_storm_info_new = df_storm_info_new.drop('track_id', axis=1)
    df_storm_info_new.insert(0, 'storm_id', None)
    
    unique_track_loop = [i for i in np.unique(all_tracks.track_id)]
    delta = 0
    for i in unique_track_loop :
        cycl=all_tracks[all_tracks.track_id==i].copy()
        cycl_info = df_storm_info[df_storm_info.track_id==i].copy()
        
        cycl_shift = cycl.copy()
        cycl_shift.lon = np.abs(cycl_shift.lon+7.5)
        cycl_lon_eu = cycl_shift.loc[cycl_shift.lon == min(cycl_shift.lon)]
        if(len(cycl_lon_eu.index)>1) :
            date_storm = datetime.datetime(year=int(cycl_lon_eu.year.iloc[0]),
                               month=int(cycl_lon_eu.month.iloc[0]),
                               day=int(cycl_lon_eu.day.iloc[0]),
                               hour=int(cycl_lon_eu.hour.iloc[0]))
            lon_storm = float(cycl_lon_eu.lon.iloc[0]-7.5)
            lat_storm = float(cycl_lon_eu.lat.iloc[0])
        else : 
            date_storm = datetime.datetime(year=int(cycl_lon_eu.year),
                                           month=int(cycl_lon_eu.month),
                                           day=int(cycl_lon_eu.day),
                                           hour=int(cycl_lon_eu.hour))
            lon_storm = float(cycl_lon_eu.loc[:,'lon']-7.5)
            lat_storm = float(cycl_lon_eu.loc[:,'lat'])
            
        storm_id_loop = str(date_storm)+'_'+str(lon_storm)+'_'+str(lat_storm)
        
        #For the track 
        cycl = cycl.drop('track_id', axis =1)
        cycl['storm_id'] = storm_id_loop
        
        all_tracks_new = pd.concat([all_tracks_new, cycl]) 
        
        #For the track info
        cycl_info = cycl_info.drop('track_id', axis =1)
        cycl_info['storm_id'] = storm_id_loop
        
        df_storm_info_new = pd.concat([df_storm_info_new, cycl_info]) 
    return all_tracks_new, df_storm_info_new

#Additional methods 
#The following additional methods are computed with the tracks identifyed with their "storm_id" and no longer their "track_id"

def concat_catalogs(df_storm_1, df_storm_info_1, df_storm_2, df_storm_info_2) :
    """
    INPUT 
        df_storm_1 : DataFrame in the format of TempestExtreme output containing all the tracks which can be considered as european storm 
            for the 1rst DataFrame
        df_storm_info_1 : DataFrame with one row per European storm. Constains general informations about the storm (surface impacted, minimal slp ....)
            for the 1rst DataFrame
        df_storm_2 : Tracks of storms (as above) for the 2nd DataFrame 
        df_storm_info_2 : Info of storms (as above) for the 2nd DataFrame 
    OUTPUT
        df_storm_all : Tracks of storms (as above) merging the two DataFrame
        df_storm_info_all : Info of storms (as above) merging the two DataFrame
    """
    df_storm_all = df_storm_1
    df_storm_info_all = df_storm_info_1
    #Loop over the track of 2nd DataFrame
    for id_loop in df_storm_info_2['storm_id'] :
        cycl_loop = df_storm_2[df_storm_2['storm_id'] == id_loop]
        cycl_info_loop = df_storm_info_2[df_storm_info_2['storm_id'] == id_loop]
        #If the track is not yet in the 1st DataFrame, we add it
        if not id_loop in df_storm_info_1['storm_id'] :
            df_storm_all = pd.concat([df_storm_all, cycl_loop], ignore_index=True)
            df_storm_info_all = pd.concat([df_storm_info_all, cycl_info_loop], ignore_index=True)
    
    return (df_storm_all, df_storm_info_all)

def SSI(df_storm_output, df_storm_info, path_data, data_name, aggregated, wind_th, geo_shp_dict, cumul_type="single", r=700, path_save_data=PATH_TRACKS, save_name="tracks_europe_info") :
    """
    INPUT :
        df_storm_output : DataFrame in the format of TempestExtreme output containing all the tracks which can be considered as european storm
        df_storm_info : DataFrame with one row per European storm. Constains general informations about the storm (surface impacted, minimal slp ....) 
        tp_dict_geo : Dictionarry of geo containing daily cumulative precipitations masked over the given geo 
        cumul_type : 
            If 'single' compute the cumul on a single track
            If 'union', compute the cumul on the union of several tracks given in the input 
            If 'inter', compute the cumul on the intersection of several tracks given in the input
    OUTPUT :
        df_storm_info : Information DataFrame with additional information about the precipitation cumul
        
    The function compute the Storm Severity Index for individual storms of for clusters
    The function load the data directly from a given path  
    """    
    geo_keys = list(geo_shp_dict.keys())
    
    #Condition the name of the key 
    if cumul_type == 'single' :
        save_key = 'SSI_'
    elif cumul_type == 'union' :
        save_key = 'SSI_union_' 
    elif cumul_type == 'inter' :
        save_key = 'SSI_inter_'
    else : 
        raise ValueError("Invalid cumul_type. Must be one of: 'single', 'union', 'inter'.")
    
    #If the keys are not present in the DataFrame, add them   
    for geo in geo_keys :
        if not save_key+geo in df_storm_info.columns :
            df_storm_info[save_key+geo]=np.nan#[np.NaN]*len(df_storm_info)
    
    #Loop over the tracks and add the values if it's a Nan
    if cumul_type=="single" :
        for tracki in np.unique(df_storm_info.storm_id) :
            df_track_loop = df_storm_output[df_storm_output.storm_id == tracki].reset_index(drop=True)
            df_info_loop = df_storm_info[df_storm_info.storm_id == tracki].reset_index(drop=True)
            #Load the needed data 
            u10_nc = extract_subset(path_data+'u10/', 'u'+data_name, df_track_loop, aggregated)
            v10_nc = extract_subset(path_data+'v10/', 'v'+data_name, df_track_loop, aggregated)
            wind = np.sqrt(u10_nc.u10**2 + v10_nc.v10**2)
            #wind_max = wind.max('time')            
            
            for geo in geo_keys :
                if np.isnan(df_info_loop[save_key+geo].values[0]):
                    #Mask the data over the shape
                    wind_copy = wind.copy()
                    wind_copy.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
                    wind_copy.rio.write_crs("epsg:4326", inplace=True)
                    wind_max_geo = wind_copy.rio.clip(geo_shp_dict[geo].geometry.values, geo_shp_dict[geo].crs)
                    
                    #Mask around the track to compute the SSI 
                    wind_masked_track = mask_around_track(wind_max_geo, df_track_loop, r)
                    wind_masked_track_max = wind_masked_track.max('time')
                    wind_masked_track_max = wind_masked_track_max.where(wind_masked_track_max>wind_th)
                    SSI_xr = (wind_masked_track_max/wind_th - 1)**3
                    SSI = SSI_xr.sum(['latitude', 'longitude'])
                    df_storm_info.loc[df_info_loop.index, save_key+geo] = float(SSI.values)

#                 #Save troughout the run 
#                 df_storm_info.to_csv(path_save_data+save_name+".csv", encoding='utf-8', index=False)
    elif cumul_type == "union" :
        for clusti in np.unique(df_storm_info.clust_id) :
            df_track_loop = df_storm_output[df_storm_output.clust_id == clusti]
            df_info_loop = df_storm_info[df_storm_info.clust_id == clusti]
            #Load the needed data 
            u10_nc = extract_subset(path_data+'u10/', 'u'+data_name, df_track_loop, aggregated)
            v10_nc = extract_subset(path_data+'v10/', 'v'+data_name, df_track_loop, aggregated)
            wind = np.sqrt(u10_nc.u10**2 + v10_nc.v10**2)
            #wind_max = wind.max('time')            
            
            for geo in geo_keys :
                if np.isnan(df_info_loop[save_key+geo].values[0]):
                    #Mask the data over the shape
                    wind_copy = wind.copy()
                    wind_copy.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
                    wind_copy.rio.write_crs("epsg:4326", inplace=True)
                    wind_max_geo = wind_copy.rio.clip(geo_shp_dict[geo].geometry.values, geo_shp_dict[geo].crs)
                    
                    #Mask around the track to compute the SSI 
                    wind_masked_track = mask_several_tracks_union(wind_max_geo, df_track_loop, r)
                    wind_masked_track_max = wind_masked_track.max('time')
                    wind_masked_track_max = wind_masked_track_max.where(wind_masked_track_max>wind_th)
                    SSI_xr = (wind_masked_track_max/wind_th - 1)**3
                    SSI = SSI_xr.sum(['latitude', 'longitude'])
                    df_storm_info.loc[df_info_loop.index, save_key+geo] = float(SSI.values)
                    
    elif cumul_type == 'inter' :
        for clusti in np.unique(df_storm_info.clust_id) :
            df_track_loop = df_storm_output[df_storm_output.clust_id == clusti]
            df_info_loop = df_storm_info[df_storm_info.clust_id == clusti]
            #Load the needed data 
            u10_nc = extract_subset(path_data+'u10/', 'u'+data_name, df_track_loop, aggregated)
            v10_nc = extract_subset(path_data+'v10/', 'v'+data_name, df_track_loop, aggregated)
            wind = np.sqrt(u10_nc.u10**2 + v10_nc.v10**2)
            #wind_max = wind.max('time')            
            
            for geo in geo_keys :
                if np.isnan(df_info_loop[save_key+geo].values[0]):
                    #Mask the data over the shape
                    wind_copy = wind.copy()
                    wind_copy.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
                    wind_copy.rio.write_crs("epsg:4326", inplace=True)
                    wind_max_geo = wind_copy.rio.clip(geo_shp_dict[geo].geometry.values, geo_shp_dict[geo].crs)
                    
                    #Mask around the track to compute the SSI 
                    wind_masked_track, wind_masked_track_indiv = mask_several_tracks_inter(wind_max_geo, df_track_loop, r)
                    wind_masked_track_max = wind_masked_track.max('time')
                    wind_masked_track_max = wind_masked_track_max.where(wind_masked_track_max>wind_th)
                    SSI_xr = (wind_masked_track_max/wind_th - 1)**3
                    SSI = SSI_xr.sum(['latitude', 'longitude'])
                    df_storm_info.loc[df_info_loop.index, save_key+geo] = float(SSI.values)            
    return(df_storm_info)  

def SSI_v2(
    df_storm_output, df_storm_info, path_data, data_name, aggregated, wind_th,
    geo_shp_dict, cumul_type="single", r=700, 
    path_save_data=PATH_TRACKS, 
    save_name="tracks_europe_info"
):
    """
    Computes the Storm Severity Index (SSI) for individual storms or clusters.
    
    Parameters:
    - df_storm_output: DataFrame with storm tracks from TempestExtremes output.
    - df_storm_info: DataFrame with information about storms or clusters.
    - path_data: Path to wind data files (u10, v10).
    - data_name: Identifier for wind data files.
    - aggregated: Boolean for data aggregation.
    - wind_th: Wind threshold for SSI computation.
    - geo_shp_dict: Dictionary with geographical regions (GeoDataFrames).
    - cumul_type: Type of SSI computation ('single', 'union', 'inter').
    - r: Radius around storm centers (in km) for impact area.
    - path_save_data: Path to save intermediate results.
    - save_name: File name for saving results.
    
    Returns:
    - df_storm_info: Updated DataFrame with computed SSI values.
    """
    geo_keys = list(geo_shp_dict.keys())

    # Validate cumul_type and define save_key
    if cumul_type == 'single':
        save_key = 'SSI_'
        group_key = 'storm_id'
    elif cumul_type == 'union':
        save_key = 'SSI_union_'
        group_key = 'clust_id'
    elif cumul_type == 'inter':
        save_key = 'SSI_inter_'
        group_key = 'clust_id'
    else:
        raise ValueError("Invalid cumul_type. Must be one of: 'single', 'union', 'inter'.")

    # Add missing SSI columns
    for geo in geo_keys:
        col_name = save_key + geo
        if col_name not in df_storm_info.columns:
            df_storm_info[col_name] = np.nan

    # Helper function to load wind data    
    def load_wind_data(track_data):
        u10_nc = extract_subset(path_data + 'u10/', 'u' + data_name, track_data, aggregated)
        v10_nc = extract_subset(path_data + 'v10/', 'v' + data_name, track_data, aggregated)
        return np.sqrt(u10_nc.u10**2 + v10_nc.v10**2)

    # Helper function to compute SSI
    def compute_ssi(wind, mask, wind_th):
        wind_masked = wind.where(mask)
        wind_masked_max = wind_masked.max('time').where(wind_masked.max('time') > wind_th)
        ssi_xr = ((wind_masked_max / wind_th - 1) ** 3).sum(['latitude', 'longitude'])
        return float(ssi_xr.values) if not ssi_xr.isnull() else np.nan

    # Loop over groups
    for group_id, group_df in df_storm_info.groupby(group_key):
        track_data = df_storm_output[df_storm_output[group_key] == group_id].reset_index(drop=True)
        wind = load_wind_data(track_data)

        for geo in geo_keys:
            col_name = save_key + geo
            if pd.isna(group_df.iloc[0][col_name]):
                if cumul_type == 'single':
                    # Single track SSI computation
                    mask = mask_around_track(wind, track_data, r)
                elif cumul_type == 'union':
                    # Union of tracks SSI computation
                    mask = mask_several_tracks_union(wind, track_data, r)
                elif cumul_type == 'inter':
                    # Intersection of tracks SSI computation
                    masks = []
                    for _, storm in group_df.iterrows():
                        storm_track = track_data[track_data.storm_id == storm.storm_id].reset_index(drop=True)
                        masks.append(mask_around_track(wind, storm_track, r))
                    mask = xr.concat(masks, dim='time').all(dim='time')
                
                # Clip wind data to geo region
                wind = wind.rio.write_crs("epsg:4326", inplace=True)
                wind_geo = wind.rio.clip(geo_shp_dict[geo].geometry.values, geo_shp_dict[geo].crs)
                
                # Compute SSI and update DataFrame
                ssi = compute_ssi(wind_geo, mask, wind_th)
                df_storm_info.loc[group_df.index, col_name] = ssi
                #Save troughout the run 
                df_storm_info.to_csv(path_save_data+save_name+".csv", encoding='utf-8', index=False)

    return df_storm_info

def SSI_wgust_v2(
    df_storm_output, df_storm_info, path_data, data_name, variable_name ,aggregated, wind_th,
    geo_shp_dict, cumul_type="single", r=700, 
    path_save_data=PATH_TRACKS, 
    save_name="tracks_europe_info"
):
    """
    Computes the Storm Severity Index (SSI) for individual storms or clusters.
    
    Parameters:
    - df_storm_output: DataFrame with storm tracks from TempestExtremes output.
    - df_storm_info: DataFrame with information about storms or clusters.
    - path_data: Path to wind data files (u10, v10).
    - data_name: Identifier for wind data files.
    - aggregated: Boolean for data aggregation.
    - wind_th: Wind threshold for SSI computation.
    - geo_shp_dict: Dictionary with geographical regions (GeoDataFrames).
    - cumul_type: Type of SSI computation ('single', 'union', 'inter').
    - r: Radius around storm centers (in km) for impact area.
    - path_save_data: Path to save intermediate results.
    - save_name: File name for saving results.
    
    Returns:
    - df_storm_info: Updated DataFrame with computed SSI values.
    """
    geo_keys = list(geo_shp_dict.keys())

    # Validate cumul_type and define save_key
    if cumul_type == 'single':
        save_key = 'SSI_'
        group_key = 'storm_id'
    elif cumul_type == 'union':
        save_key = 'SSI_union_'
        group_key = 'clust_id'
    elif cumul_type == 'inter':
        save_key = 'SSI_inter_'
        group_key = 'clust_id'
    else:
        raise ValueError("Invalid cumul_type. Must be one of: 'single', 'union', 'inter'.")

    # Add missing SSI columns
    for geo in geo_keys:
        col_name = save_key + geo
        if col_name not in df_storm_info.columns:
            df_storm_info[col_name] = np.nan

    # Helper function to compute SSI
    def compute_ssi(wind, mask, wind_th):
        wind_masked = wind.where(mask)
        wind_masked_max = wind_masked.max('time').where(wind_masked.max('time') > wind_th)
        ssi_xr = ((wind_masked_max / wind_th - 1) ** 3).sum(['latitude', 'longitude'])
        return float(ssi_xr.values) if not ssi_xr.isnull() else np.nan

    # Loop over groups
    for group_id, group_df in df_storm_info.groupby(group_key):
        track_data = df_storm_output[df_storm_output[group_key] == group_id].reset_index(drop=True)
        wind_nc = extract_subset(path_data , data_name, track_data, aggregated)
        wind = wind_nc[variable_name]

        for geo in geo_keys:
            col_name = save_key + geo
            if pd.isna(group_df.iloc[0][col_name]):
                if cumul_type == 'single':
                    # Single track SSI computation
                    mask = mask_around_track(wind, track_data, r)
                elif cumul_type == 'union':
                    # Union of tracks SSI computation
                    mask = mask_several_tracks_union(wind, track_data, r)
                elif cumul_type == 'inter':
                    # Intersection of tracks SSI computation
                    masks = []
                    for _, storm in group_df.iterrows():
                        storm_track = track_data[track_data.storm_id == storm.storm_id].reset_index(drop=True)
                        masks.append(mask_around_track(wind, storm_track, r))
                    mask = xr.concat(masks, dim='time').all(dim='time')
                
                # Clip wind data to geo region
                wind = wind.rio.write_crs("epsg:4326", inplace=True)
                wind_geo = wind.rio.clip(geo_shp_dict[geo].geometry.values, geo_shp_dict[geo].crs)
                
                # Compute SSI and update DataFrame
                ssi = compute_ssi(wind_geo, mask, wind_th)
                df_storm_info.loc[group_df.index, col_name] = ssi
                #Save troughout the run 
                df_storm_info.to_csv(path_save_data+save_name+".csv", encoding='utf-8', index=False)

    return df_storm_info

#Additional in between functions 

def storm_dates_tracking(df_all_tracks) :
    """
    INPUT 
        df_all_tracks : DataFrame in the format of TempestExtreme output containing all the tracks which can be considered as european storm
    OUTPUT 
        List of dates in datetime.datetime format correponding to the storm landing date
    """
    dates = []
    #track_id_list = np.unique(df_all_tracks.track_id)
    storm_id_list = np.unique(df_all_tracks.storm_id)
    for id_t in storm_id_list :
        cycl = df_all_tracks.loc[df_all_tracks.storm_id == id_t].copy()
        cycl.lon = np.abs(cycl.lon+7.5)
        cycl_lon_eu = cycl.loc[cycl.lon == min(cycl.lon)]
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
        dates.append(date_storm)
    return(dates)

#The function below mask the grid points which are further than a radius around the center of minimal pressure 

def safe_clip(input_copy, gdf_mask):
    # Check if there's any data in bounds before clipping
    try:
        # Attempt to clip and handle exceptions if no data is found
        input_copy_mask = input_copy.rio.clip(gdf_mask.geometry, gdf_mask.crs)
        return input_copy_mask
    except rioxarray.exceptions.NoDataInBounds:
        #print("No data found within the specified bounds. Returning an empty DataArray or handling it appropriately.")
        # You could return None or an empty xarray DataArray, depending on your needs
        empty_data = xr.DataArray(
                            data=np.full_like(input_copy, np.nan),  
                            coords=input_copy.coords,  
                            dims=input_copy.dims,  
                            attrs=input_copy.attrs)
        return empty_data

def mask_around_track(input_xr, track_df, r=700) : 
    """
    INPUT 
        input_xr : Xarray of all the timesteps and the selected variable 
        track_df : Individual track in DataFrame format (TempestExtreme)
        r : Radius of influence around the minimum of pressure
    OUTPUT
        DataArray with NaN values outside the area of impact and sliced temporally at at the days of the trajectory. 
        This function can only be used for an individual track
    """
    delta_t = datetime.timedelta(hours=12)
    first_date = datetime.datetime(year=int(track_df.iloc[0].year),
                                   month=int(track_df.iloc[0].month),
                                   day=int(track_df.iloc[0].day), 
                                  hour=int(track_df.iloc[0].hour)) - delta_t
    last_date  = datetime.datetime(year=int(track_df.iloc[-1].year),
                                   month=int(track_df.iloc[-1].month),
                                   day=int(track_df.iloc[-1].day), 
                                  hour=int(track_df.iloc[-1].hour)) + delta_t
    
    #Open the xarray and make sure it has crs set
    input_copy = input_xr.sel(time=slice(first_date, last_date)).load().copy() 
    input_copy = input_copy.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
    input_copy = input_copy.rio.write_crs("EPSG:4326", inplace=True)
    
    input_copy_crop = input_copy.copy()
    
    output = xr.zeros_like(input_copy)
    
    for id_t in track_df.index :
        track_at_time = track_df.loc[[id_t], :]
        time_i = datetime.datetime(year=int(track_at_time.year),
                               month=int(track_at_time.month),
                               day=int(track_at_time.day), 
                              hour=int(track_at_time.hour))
        
        #If the loop is over the last point, select the xarray until the next 6 hours
        if (id_t == track_df.index[-1]):
            time_next = time_i+datetime.timedelta(hours=12)
        
        #Else, look directly for the next time
        else : 
            tracks_at_time_next_point = track_df.loc[id_t+1, :]
            time_next = datetime.datetime(year=int(tracks_at_time_next_point.year),
                                           month=int(tracks_at_time_next_point.month),
                                           day=int(tracks_at_time_next_point.day), 
                                          hour=int(tracks_at_time_next_point.hour))
        
        #Prepare point mask
        gdf = gpd.GeoDataFrame(track_at_time,
                               geometry = gpd.points_from_xy(track_at_time.lon, track_at_time.lat),
                               crs="EPSG:4326")
        mask = gdf.to_crs(epsg=3395).buffer(1000*r)
        gdf_mask = mask.to_crs(epsg=4326)
        
        #Mask the input file 
        input_copy_mask       = input_copy_crop.sel(time=slice(time_i-delta_t, time_next+delta_t)).copy()
#         input_copy_mask       = input_copy.rio.clip(gdf_mask.geometry, gdf_mask.crs)
        
        try:
            input_copy_mask = input_copy_mask.rio.clip(gdf_mask.geometry, gdf_mask.crs)
        except rioxarray.exceptions.NoDataInBounds:
            #print("Warning: No data found in bounds after clipping. Returning an empty dataset.")
            input_copy_mask = xr.zeros_like(input_copy_mask)  # Return an empty dataset instead of failing
        
    
        input_copy_mask_align = input_copy_mask.reindex(
            latitude=input_copy.latitude, 
            longitude=input_copy.longitude
        )
        
        if id_t == track_df.index[0]: 
            output = input_copy_mask_align.sel(time=slice(time_i-delta_t, time_next+delta_t))
        else:             
            output = xr.concat([output, input_copy_mask_align.sel(time=slice(time_i-delta_t, time_next+delta_t))], dim="time")
            
    return output
        
def mask_several_tracks_inter(input_xr, tracks_df, r=700) :
    """
    INPUT 
        input_xr : Raw netcdf over which the footprint will be constructed
        list_tracks_df : DataFrame (output of TempestExtreme) which contains several individual tracks 
        r : Radius of influence around the minimum of pressure
    OUTPUT 
        The function return a footprint corresponding to the intersection of individual storm footprints. 
        The output file as a format (time, longitude, latitude) with NaN values for grid point
        which are not in the impact area of ALL of the storms
    """
    list_tracki = np.unique(tracks_df.storm_id)
    masked_field = []
    for tracki in list_tracki :
        track_loop = tracks_df.loc[tracks_df.storm_id == tracki]
        masked_loop = mask_around_track(input_xr, track_loop, r)
        masked_field.append(masked_loop) 
    
    #Create the mask to apply to get the intersection 
    if len(masked_field)==1 :
        input_xr_inter = masked_field[0].max('time')
    elif len(masked_field)==2 : 
        input_xr_inter = np.maximum(masked_field[0].max('time'), masked_field[1].max('time'))
    elif len(masked_field)!=0: 
        input_xr_inter = np.maximum(masked_field[0].max('time'), masked_field[1].max('time'))
        #del masked_field[0:2]
        index = 2
        while index<len(masked_field) :
            input_xr_inter = np.maximum(input_xr_inter, masked_field[index].max('time'))
            index += 1
            
    #Apply the mask over the individual tracks 
    for id_tracki, tracki in enumerate(list_tracki) : 
        new_masked = masked_field[id_tracki].where([np.logical_not(np.isnan(input_xr_inter))]*len(masked_field[id_tracki].time))
        masked_field[id_tracki] = new_masked
    
    #Compute the union and apply mask over it 
    masked_field_union = mask_several_tracks_union(input_xr, tracks_df, r)
    masked_field_inter = masked_field_union.where([np.logical_not(np.isnan(input_xr_inter))]*len(masked_field_union.time))
    
    return(masked_field_inter, masked_field)

def mask_several_tracks_union(input_xr, tracks_df, r=700) :
    """
    INPUT 
        input_xr : Raw netcdf over which the footprint will be constructed
        tracks_df : DataFrame (output of TempestExtreme) which contains several individual tracks
        r : Radius of influence around the minimum of pressure
    OUTPUT 
        The function return a footprint corresponding to the union of individual storm footprints. 
        The output file as a format (time, longitude, latitude) with NaN values for grid point
        which are not in the impact area of ANY of the storms
    """
    #Find the first and last dates 
    list_first_date = []
    list_last_date = []
    for tracki in np.unique(tracks_df.storm_id) :
        track_loop = tracks_df.loc[tracks_df.storm_id == tracki]
        #Select the first date of each track
        date_first_loop = datetime.datetime(year=int(track_loop.iloc[0].year),
                                   month=int(track_loop.iloc[0].month),
                                   day=int(track_loop.iloc[0].day), 
                                  hour=int(track_loop.iloc[0].hour))
        list_first_date.append(date_first_loop)
        
        #Select the last date of each track 
        date_last_loop = datetime.datetime(year=int(track_loop.iloc[-1].year),
                           month=int(track_loop.iloc[-1].month),
                           day=int(track_loop.iloc[-1].day), 
                          hour=int(track_loop.iloc[-1].hour))+datetime.timedelta(hours=6)
        list_last_date.append(date_last_loop)
    first_date = min(list_first_date)
    last_date =  max(list_last_date)
    
    #Create the xarray
    input_copy = input_xr.sel(time=slice(first_date, last_date))
    
    for id_s, stormi in enumerate(np.unique(tracks_df.storm_id)) :
        storm_loop = tracks_df.loc[tracks_df.storm_id == stormi]
        masked_loop = mask_around_track(input_xr, storm_loop)
        
        # Create an empty array 
        masked_array = xr.DataArray(
                            data=np.full_like(input_copy, np.nan),  
                            coords=input_copy.coords,  
                            dims=input_copy.dims,  
                            attrs=input_copy.attrs)
        masked_array.loc[dict(time=slice(masked_loop.time[0], masked_loop.time[-1]))] = masked_loop
        masked_array.assign_coords(storm=id_s)
        
        if id_s == 0 : 
            input_union = masked_array
        else : 
            input_union = xr.concat([input_union, masked_array], dim='ensemble')
    input_union = input_union.max('ensemble')
    return(input_union)
    
# Following function define the storm based on some exceeded threshold 
def surf_above_th(input_xr, threshold, is_mask=None, mask=None, delta_lat=0.25, delta_lon=0.25) :
    """
    INPUT 
        input_xr : 2d xarray field 
        threshold : 2d xarray of the threshold field 
        mask : a shapefile of the region that needs to be masked 
    OUTPUT
        Return the surface (in km²) above a xarray 2d threshold
    """
    diff = input_xr - threshold
    
    #Replace point where the threshold is not over passed by a Nan
    diff_with_Nan = diff.where(diff>0, np.nan)
    
    if not is_mask : 
        diff_with_Nan_masked = diff_with_Nan.copy()
    else : 
        #Mask
        diff_with_Nan_copy = diff_with_Nan.copy()
        diff_with_Nan_copy.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
        diff_with_Nan_copy.rio.write_crs("epsg:4326", inplace=True)
        diff_with_Nan_masked = diff_with_Nan_copy.rio.clip(mask.geometry.values, mask.crs)
        
    count_above = len(diff_with_Nan_masked.longitude) - np.isnan(diff_with_Nan_masked).sum('longitude')
    surf_above = 0
    for np_cell, lat_i in zip(count_above, diff_with_Nan_masked.latitude):
        surf_above += np_cell*np.radians(delta_lon)*6371**2*np.abs(np.sin(np.radians(lat_i+delta_lat/2)) - np.sin(np.radians(lat_i-delta_lat/2)))
    return(float(surf_above))

#Function to look for the year and month we want to extract
#Do not import all the hourly data but ony a subset 
def extract_subset(path_data, data_name, tracks, aggregated='ym'):
    """
    INPUT : 
        path_data
        data_name
        tracks
        aggregated : If 'ym', need to use both year and month the extract need files 
        If 'y', need only the year 
    OUTPUT :
        Import a restricted portion of divided netcdf files 
    """
    y_min = min(tracks.year)
    y_max = max(tracks.year)
    files_paths = []
    if aggregated == 'y' :
        for y in range(y_min, y_max+1, 1) :
            files_paths.append(path_data+data_name+'_'+str(y)+'.nc')
    elif aggregated == 'ym' :
        if y_min != y_max :
            tracks_ymin = tracks.loc[tracks.year == y_min]
            tracks_ymax = tracks.loc[tracks.year == y_max]
            for month in np.unique(tracks_ymin.month) :
                files_paths.append(path_data+data_name+'_'+str(y_min)+str(month).zfill(2)+'.nc')
            for month in np.unique(tracks_ymax.month) :
                files_paths.append(path_data+data_name+'_'+str(y_max)+str(month).zfill(2)+'.nc')
        else : 
            for month in np.unique(tracks.month) :
                files_paths.append(path_data+data_name+'_'+str(y_min)+str(month).zfill(2)+'.nc')
    data = xr.open_mfdataset(files_paths, combine='by_coords', 
                             parallel=True, chunks={"lat": 100, "lon": 200}, 
                             coords='minimal', compat='override')#, combine='nested', concat_dim='time')
    if "valid_time" in data:
        data = data.rename({"valid_time": "time"})
    return data 

#### FUNCTIONS TO COMPUTE FOOTPRINT

def footprint_shp(tracks_df, path_data, data_name, variable_name, aggregated,
                  gather, path_save, is_mask=None, mask=None, r=700):
    """
    INPUT 
        aggregated : 'ym', 'y' information about the way variable to extract are aggregated in files
        gather : The method of gathering the timesteps (mean, max, median, .... )
        r : Radius of influence around the minimum of pressure
    OUTPUT 
        Create the footprint over a given variable and save it to a shape file and netcdf 
    """
    storm_ids = np.unique(tracks_df.storm_id)
    for stormi in storm_ids :
        df_track_loop = tracks_df[tracks_df.storm_id == stormi]
        #Extract only the needed years
        data_nc = extract_subset(path_data, data_name, tracks_df, aggregated)
        data_nc = data_nc[variable_name]
        
        #Mask over a given domain 
        if not is_mask : 
            data_nc_masked = data_nc.copy()
        else : 
            #Mask
            diff_with_Nan_copy = data_nc.copy()
            diff_with_Nan_copy.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
            diff_with_Nan_copy.rio.write_crs("epsg:4326", inplace=True)
            data_nc_masked = diff_with_Nan_copy.rio.clip(mask.geometry.values, mask.crs)
        
        #Mask around the track 
        data_nc_masked_track = mask_around_track(data_nc_masked, df_track_loop, r)
        
        #Compute the needed gathering of timesteps 
        if gather == 'max' : 
            data_nc_masked_track_gather = data_nc_masked_track.max('time')
        elif gather == 'mean' :
            data_nc_masked_track_gather = data_nc_masked_track.mean('time')
        elif gather == 'median' :
            data_nc_masked_track_gather = data_nc_masked_track.median('time')
        else : 
            return ('gather not valid')
        
        lat, lon = xr.broadcast(data_nc_masked_track_gather.longitude, data_nc_masked_track_gather.latitude)
        geometry = [Point(lat_val, lon_val) for lat_val, lon_val in zip(lat.values.flatten(), lon.values.flatten())]
        gdf = gpd.GeoDataFrame(data_nc_masked_track_gather.transpose().values.flatten(), geometry=geometry, columns=['value'], crs="EPSG:4326")
        
        #Save shp file 
        gdf.to_file(path_save+stormi+'_'+gather+'.shp', driver='ESRI Shapefile')
        
    return (data_nc_masked_track_gather, gdf)

def footprint_nc(tracks_df, track_df_info, path_data, data_name, variable_name, aggregated,
                  gather, save = True, path_save=PATH_FOOTPRINTS, is_mask=None, mask=None, r=900):
    """
    INPUT 
        aggregated : 'ym', 'y' information about the way variable to extract are aggregated in files
        gather : The method of gathering the timesteps (mean, max, median, .... )
        r : Radius of influence around the minimum of pressure
    OUTPUT 
        Create the footprint over a given variable and save it to a shape file and netcdf 
    """
    storm_ids = np.unique(tracks_df.storm_id)
    for stormi in storm_ids :
        df_track_loop = tracks_df[tracks_df.storm_id == stormi].reset_index(drop=True)
        df_info_track_loop = track_df_info[track_df_info.storm_id == stormi]
        ## Extract only the needed years
        data_nc = extract_subset(path_data, data_name, df_track_loop, aggregated)
        data_nc = data_nc[variable_name]
        
        ## Mask over a given domain 
        if not is_mask : 
            data_nc_masked = data_nc.copy()
        else : 
            ## Mask
            diff_with_Nan_copy = data_nc.copy()
            diff_with_Nan_copy.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
            diff_with_Nan_copy.rio.write_crs("epsg:4326", inplace=True)
            data_nc_masked = diff_with_Nan_copy.rio.clip(mask.geometry.values, mask.crs)
        
        ## Mask around the track 
        data_nc_masked_track = mask_around_track(data_nc_masked, df_track_loop, r)
        
        ## Compute the needed gathering of timesteps 
        if gather == 'max' : 
            data_nc_masked_track_gather = data_nc_masked_track.max('time')
        elif gather == 'mean' :
            data_nc_masked_track_gather = data_nc_masked_track.mean('time')
        elif gather == 'median' :
            data_nc_masked_track_gather = data_nc_masked_track.median('time')
        else : 
            return ('gather not valid')
         
        ## Add a time dimension corresponding to the landing date 
        data_nc_masked_track_gather = data_nc_masked_track_gather.expand_dims(time=df_info_track_loop.storm_landing_date)
        
        ## Convert to Dataset and Add attributes        
        new_name = f"{gather}_{variable_name}"
#         dataset = data_nc_masked_track_gather.to_dataset(name='max_wind_gust')
        dataset = data_nc_masked_track_gather.to_dataset(name=new_name)
        #dataset.attrs['ssi']                       = str(df_info_track_loop['SSI_FRA'].iloc[0])
        dataset.attrs['ssi_spatial_extend']        = 'FRA'
        dataset.attrs['mean_gust_speed25']         = float(data_nc_masked_track_gather.mean())
        dataset.attrs['mean_gust_speed25_unit']    = 'm/s'
        dataset.attrs['max_gust_speed']            = float(data_nc_masked_track_gather.max())
        dataset.attrs['max_gust_speed_unit']       = 'm/s'
        dataset.attrs['comment']                   = 'um-version = Euro4 downscaler'
        dataset.attrs['data_type']                 = 'grid'
        dataset.attrs['creator_email']             = 'laura.hasbini@lsce.ipsl.fr'
        dataset.attrs['product_version']           = '1.0'
        dataset.attrs['geospatial_lat_min']        = float(data_nc_masked_track_gather.latitude.min())
        dataset.attrs['geospatial_lat_resolution'] = np.abs(float(data_nc_masked_track_gather.latitude[1])-float(data_nc_masked_track_gather.latitude[0]))
        dataset.attrs['geospatial_lat_max']        = float(data_nc_masked_track_gather.latitude.max())
        dataset.attrs['geospatial_lon_min']        = float(data_nc_masked_track_gather.longitude.min())
        dataset.attrs['geospatial_lon_resolution'] = np.abs(float(data_nc_masked_track_gather.longitude[1])-float(data_nc_masked_track_gather.longitude[0]))
        dataset.attrs['geospatial_lon_max']        = float(data_nc_masked_track_gather.longitude.max())
        dataset.attrs['time_coverage_start']       = str(track_df_info['storm_landing_date'])
        dataset.attrs['geospatial_lat_units']      = 'degrees_north'
        dataset.attrs['geospatial_lon_units']      = 'degrees_east'
        dataset.attrs['keywords']                  = 'wind storm footprints, storm tracks'
        dataset.attrs['storm_name']                = stormi
        
        ## Save netcdf
        if save : 
            dataset.to_netcdf(path_save+stormi+'_'+gather+'_r'+str(r)+'.nc')
    return (dataset)

#### SSI from footprint
def SSI_from_footprint(df_info_tracks, new_col_name,
                       path_footprint = PATH_FOOTPRINTS, variable_name='max_wind_gust', r=1300, 
                       wind_th_nc=None, 
                       is_mask = False, mask=None, 
                       save_during = False, save_path = PATH_TRACKS, save_name = "tracks_info_SSI_test", 
                       exposure = False, exposure_data = None, exposure_var = None
                       ) : 
    df_info_tracks[new_col_name] = np.nan
    
    for stormi in df_info_tracks.storm_id.unique() : 
        stormi_path = os.path.join(path_footprint, stormi+"_max_r"+str(r)+".nc" )
        ds = xr.open_mfdataset(stormi_path,
                               engine="netcdf4").drop_vars('time', errors='ignore')
    
        if 'expver' in ds.coords :
            ds = ds.sel(expver=1)
        
        if is_mask : 
            # Filter with the mask 
            ds_mask = ds.copy()
            ds_mask.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
            ds_mask.rio.write_crs("epsg:4326", inplace=True)
            ds_mask = ds_mask.rio.clip(mask.geometry.values, mask.crs)

            #Extract the wind only 
            ds_mask = ds_mask[variable_name]#.sel(expver=1)
        else : 
            ds_maks = ds[variable_name]#.sel(expver=1)

        #Compute the SSI 
        ds_above_th = ds_mask.where(ds_mask>wind_th_nc, np.nan)       
        ds_SSI = (ds_above_th/wind_th_nc - 1)**3
        
        if exposure : 
            ds_SSI = ds_SSI*exposure_data[exposure_var]
        SSI_stormi = float(ds_SSI.sum().values)
        
        # Store the SSI value in the DataFrame
        df_info_tracks.loc[df_info_tracks.storm_id == stormi, new_col_name] = SSI_stormi
        
        if save_during :
            save_filepath = os.path.join(save_path, f"{save_name}.csv")
            df_info_tracks.to_csv(save_filepath, encoding='utf-8', index=False)
        
    return df_info_tracks