import numpy as np
import pandas as pd
import xarray as xr
import datetime
import rioxarray 
import pytz

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from rechunker import rechunk
from rasterio import features
from affine import Affine

import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import Point
from scipy.signal import find_peaks

import dask.dataframe as dd
import multiprocessing as mp
#npartitions = max(1, mp.cpu_count() // 4)
npartitions = 8

import sys 
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from paths import *

## Dict of columns types
DTYPE_SINCLIM = {'cod_sin': str, 
                 "cod_pol" : str, 
                 "cod_cie" : str, 
                 "cod_ent" : str, 
                 "cod_res" : str, 
                 "lib_res" : str, 
                 "lib_eta" : str, 
                 "lib_lob" : str, 
                 "lib_lob2" : str, 
                 "lib_per" : str, 
                 "num_chg_brut" : float, 
                 "cod_csd" : str, 
                 "cod_reg" : str, 
                 "cod_bil" : str, 
                 "cod_nat" : str, 
                 "cod_cbo" : str, 
                 "cod_iso" : str, 
                 "lib_geo" : str, 
                 "num_lat" : float, 
                 "num_lon" : float, 
                 "cod_ins" : str, 
                 "cod_hex" : str, 
                 "cod_bat" : str, 
                 "cod_bnb" : str, 
                 "lib_bat" : str, 
                 "lib_etg" : str, 
                 "lib_occ" : str, 
                 "lib_log" : str, 
                 "lib_naf" : str, 
                 "cod_risque_ino" : str, 
                 "cod_risque_rga" : str, 
                 "lib_ver" : str}

## Preprocess the claims to focus on MRH, type_claim... 
def claims_preprocess(sinclim, type_claim, lob=False) : 
    """
    type_claim : Should be a list 
    """
    # Remove claims not linked to the actual hazard
    sinclim            = sinclim.loc[sinclim.lib_eta != 'sans_suite']
    # Remove negative damage and serious damage 
    sinclim            = sinclim.loc[sinclim.num_chg_brut > 0]
    sinclim            = sinclim.loc[sinclim.num_chg_brut < 150000]
    #Filter over the peril of interest
    sinclim            = sinclim[sinclim.lib_per.isin(type_claim)] 
    #Filter over France
    sinclim            = sinclim[sinclim.num_lat > 30]
    #Filter of GIARD entity
    sinclim            = sinclim.loc[sinclim.cod_ent=='GIARD']
#     sinclim            = sinclim.loc[sinclim.lib_lob.isin(["mri", "mrh", "mrc"])]
    #Clean the date notation
    sinclim['dat_sin'] = pd.to_datetime(sinclim['dat_sin'])
    sinclim            = sinclim[sinclim.dat_sin.dt.month.isin([1, 2, 3, 4, 9, 10, 11, 12])]
    #Convert the cod_sin to string 
    sinclim['cod_sin'] = sinclim['cod_sin'].astype(str).str.strip()
    #Filter over a given lob, if required 
    if lob : 
        sinclim        = sinclim.loc[sinclim.lib_lob==lob]
    return sinclim 

## Compute distance between point and track
from math import radians, sin, cos, sqrt, atan2
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # Radius of the Earth in kilometers
    R = 6371.0
    
    # Calculate the distance
    distance = R * c
    
    return distance

##### Change date for local maximum 
def shift_local_max(sinclim_pd) : 
    sinclim_pd['dat_sin'] = pd.to_datetime(sinclim_pd['dat_sin'])

    # Aggregate claims by date
    dat_sin = sinclim_pd['dat_sin'].value_counts().sort_index()
    dat_sin = dat_sin.asfreq('D').fillna(0)  # Fill in missing dates for continuity

    # Identify peaks in the aggregated claim data
    peaks, _ = find_peaks(dat_sin.values, height=5)  # Adjust height as needed
    peak_dates = dat_sin.index[peaks]

    # Function to find the nearest peak date for each claim
    def find_nearest_peak(date, peak_dates):
        return min(peak_dates, key=lambda peak: abs(peak - date))

    # Apply the nearest peak function to each claim's date
    sinclim_pd['new_dat_sin'] = sinclim_pd['dat_sin'].apply(find_nearest_peak, peak_dates=peak_dates)
    sinclim_pd = sinclim_pd.drop(['dat_sin'], axis=1)
    sinclim_pd = sinclim_pd.rename(columns={"new_dat_sin": "dat_sin"})    
    return sinclim_pd

def shift_local_max_dask(sinclim_pd):
    # Convert to Dask DataFrame
    sinclim_pd_dask = dd.from_pandas(sinclim_pd, npartitions=npartitions)

    # Ensure datetime conversion
    sinclim_pd_dask['dat_sin'] = dd.to_datetime(sinclim_pd_dask['dat_sin'])

    # Aggregate claims by date
    dat_sin = sinclim_pd_dask['dat_sin'].value_counts().compute().sort_index()
    dat_sin = dat_sin.asfreq('D').fillna(0)  # Ensure continuity

    # Identify peaks
    peaks, _ = find_peaks(dat_sin.values, height=5)  # Adjust threshold as needed
    peak_dates = dat_sin.index[peaks]

    # Function to find the nearest peak for each row
    def find_nearest_peak(df, peak_dates):
        df["new_dat_sin"] = df["dat_sin"].apply(lambda date: min(peak_dates, key=lambda peak: abs(peak - date)))
        return df

    # Apply function in parallel
    sinclim_pd_dask = sinclim_pd_dask.map_partitions(find_nearest_peak, peak_dates=peak_dates, meta={"dat_sin": "datetime64[ns]", "new_dat_sin": "datetime64[ns]"})

    # Drop old column and rename
    sinclim_pd_dask = sinclim_pd_dask.drop(columns=["dat_sin"]).rename(columns={"new_dat_sin": "dat_sin"})

    return sinclim_pd_dask

###### Stationnary method using only the date 
def link_claims_storm_dates(sinclim_pd, df_storm, df_storm_info, threshold_time_max=datetime.timedelta(hours=96), threshold_time_min=datetime.timedelta(hours=0)) :
    sinclim_storm = pd.DataFrame(columns = sinclim_pd.columns)
    date_landing_loop = df_storm_info.storm_landing_date.iloc[0]
    stop_after = False
    
    ### EXTRACT INFO ABOUT THE STORM
    date_storm = df_storm_info.storm_landing_date.iloc[0]
    
    ### LOOP OVER THE CLAIMS 
    for i in range(len(sinclim_pd)) :
        diff = sinclim_pd.iloc[i]['dat_sin'] - date_storm.to_pydatetime()#.astimezone(pytz.UTC)
        if diff < threshold_time_max and diff > threshold_time_min:
            sinclim_storm = sinclim_storm._append(sinclim_pd.iloc[i], ignore_index=True)
    return sinclim_storm


##### Link claims to storm candidates based on a given interval windows
def assoc_storm_candidates(sinclim_pd, d_before, d_after, df_storm, df_info_storm, update=False):
    '''
    if update==True : Do not proceed to the full association from scratch, use back the results from the previous association 
    '''
    delta_t_after = datetime.timedelta(hours=d_after*24)
    delta_t_before = datetime.timedelta(hours=-d_before*24)

    # Initialize empty list to collect DataFrames
    sinclim_storm_list = []

    for stormi in np.unique(df_info_storm.storm_id):
        df_storm_loop = df_storm[df_storm.storm_id == stormi]
        df_info_storm_loop = df_info_storm[df_info_storm.storm_id == stormi]
        
        if update:
            sinclim_subset = sinclim_pd.loc[sinclim_pd.storm_id == stormi]
            sinclim_storm_loop = link_claims_storm_dates(sinclim_subset, df_storm_loop, df_info_storm_loop, 
                                                      delta_t_after, 
                                                      delta_t_before)
        else:
            sinclim_storm_loop = link_claims_storm_dates(sinclim_pd, df_storm_loop, df_info_storm_loop, 
                                                      delta_t_after, 
                                                      delta_t_before)
        
        if not sinclim_storm_loop.empty:
            sinclim_storm_loop['storm_id'] = stormi
            # Reset index to avoid duplicate indices
            sinclim_storm_loop = sinclim_storm_loop.reset_index(drop=True)
            sinclim_storm_list.append(sinclim_storm_loop)
    
    # Concatenate all DataFrames at once
    if sinclim_storm_list:
        sinclim_storm = pd.concat(sinclim_storm_list, ignore_index=True)
    else:
        sinclim_storm = pd.DataFrame(columns=list(sinclim_pd.columns)+['storm_id'])
    
    return sinclim_storm


##### Add the windgust
def import_wgust_footprint(sinclim_storm, r=1300, path_footprint_wgust=PATH_FOOTPRINTS) : 
    # Initialize an empty list to store the datasets for each storm
    datasets = []
    datasets_storm = []
    
    # # Loop through each storm in the stormi_impact list
    for stormi in np.unique(sinclim_storm.storm_id):
        # Condition of the landing date is earlier than 2010 : 
        date_stormi_loop = datetime.datetime.strptime(stormi[:19], "%Y-%m-%d %H:%M:%S")
        
        stormi_path = os.path.join(path_footprint_wgust, stormi+"_max_r"+str(r)+".nc" )

        # Open all NetCDF files in the 'stormi_path' folder that match the pattern '*_max.nc'
        ds = xr.open_mfdataset(stormi_path,
                               engine="netcdf4").drop_vars('time', errors='ignore')

        # Append the opened dataset to the list
        datasets.append(ds)

        #Remove time dimension and change it to storm 
        ds = ds.drop_vars('time', errors='ignore')  
        ds = ds.expand_dims({'storm_id': [stormi]})
        datasets_storm.append(ds)

    # If you want to merge all the datasets into one, use xr.concat or xr.combine_by_coords
    combined_dataset_storm = xr.concat(datasets_storm, dim='storm_id')    
    return combined_dataset_storm

def import_wgust_footprint_varying_radius(sinclim_storm, path_footprint_wgust=PATH_FOOTPRINTS_VARYING_RADIUS) : 
    # Initialize an empty list to store the datasets for each storm
    datasets = []
    datasets_storm = []
    
    # # Loop through each storm in the stormi_impact list
    for stormi in np.unique(sinclim_storm.storm_id):
        # Condition of the landing date is earlier than 2010 : 
        date_stormi_loop = datetime.datetime.strptime(stormi[:19], "%Y-%m-%d %H:%M:%S")
        
        stormi_path = os.path.join(path_footprint_wgust, stormi+"_max_r*" )

        # Open all NetCDF files in the 'stormi_path' folder that match the pattern '*_max.nc'
        ds = xr.open_mfdataset(stormi_path,
                               engine="netcdf4").drop_vars('time', errors='ignore')

        # Append the opened dataset to the list
        datasets.append(ds)

        #Remove time dimension and change it to storm 
        ds = ds.drop_vars('time', errors='ignore')  
        ds = ds.expand_dims({'storm_id': [stormi]})
        datasets_storm.append(ds)

    # If you want to merge all the datasets into one, use xr.concat or xr.combine_by_coords
    combined_dataset_storm = xr.concat(datasets_storm, dim='storm_id')    
    return combined_dataset_storm

def add_wgust_new(sinclim_pd, combined_dataset_storm) : 
    sinclim_pd_dask = dd.from_pandas(sinclim_pd, npartitions=npartitions)
    def get_wgust(row):
        try:
            return float(
                combined_dataset_storm.max_wind_gust
                .sel(storm_id=row.storm_id)
                .sel(latitude=row.num_lat, longitude=row.num_lon, method='nearest')
                .values
            )
        except (KeyError, ValueError, AttributeError):
            return np.nan
    sinclim_pd_dask['wgust_max'] = sinclim_pd_dask.map_partitions(lambda df: df.apply(get_wgust, axis=1))
    return sinclim_pd_dask


def get_wgust(row, combined_dataset_storm):
    """Extract max_wind_gust for a given storm and location."""
    try:
        return float(
            combined_dataset_storm.max_wind_gust
            .sel(storm_id=row["storm_id"])
            .sel(latitude=row["num_lat"], longitude=row["num_lon"], method="nearest")
            .values
        )
    except (KeyError, ValueError, AttributeError):
        return np.nan

def add_wgust_new_parallel(sinclim_pd, combined_dataset_storm, max_workers=None, show_progress=True):
    """
    Adds a 'wgust_max' column to sinclim_pd by extracting max_wind_gust from combined_dataset_storm
    for each storm_id and nearest (lat, lon) point, in parallel.
    """
    
    rows = sinclim_pd.to_dict(orient="records")
    results = [None] * len(rows)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(get_wgust, row, combined_dataset_storm): i
            for i, row in enumerate(rows)
        }

        iterator = tqdm(as_completed(futures), total=len(futures), desc="Computing wgust_max") if show_progress else as_completed(futures)

        for future in iterator:
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception:
                results[i] = np.nan
    
    sinclim_pd = sinclim_pd.copy()
    sinclim_pd["wgust_max"] = results
    return sinclim_pd

##### Postprocessing to have feasible linking 
def gather_claims_storm(sinclim_pd, df_storm, df_storm_info, nb_min_claims = 10) : 
    """
    If a storm is linked to less than nb_min_claims, the claims associate to this storm are linked to another one. 
    Dates of df_storm_info must already be in datetime 
    Loop from the end of the file 
    Look only for the storms with a date before the claim date
    """
    sinclim_copy = sinclim_pd.copy()
    stormi_claims = np.unique(sinclim_copy.storm_id)
    #print(df_storm_info.loc[df_storm_info.isin(stormi_claims)])
    stormi_landing_dates = df_storm_info.loc[df_storm_info.storm_id.isin(stormi_claims)]['storm_landing_date']
    #pd.to_datetime(df_cluster_info['storm_landing_date'])
    
    #Identify claims to re-attribute
    sinclim_grp = sinclim_copy.groupby('storm_id').agg(n_claims = ('cod_sin', 'count'))
    stormi_rea = sinclim_grp.loc[sinclim_grp.n_claims<nb_min_claims].index.tolist()
    sinclim_rea = sinclim_pd.loc[sinclim_pd.storm_id.isin(stormi_rea)]
    
    stormi_keep = [value for value in stormi_claims if value not in stormi_rea]
    stormi_landing_dates_keep = df_storm_info.loc[df_storm_info.storm_id.isin(stormi_keep)]['storm_landing_date']
    
    #Re-attribute to the correct storm 
    sinclim_rea = sinclim_rea.sort_values(by='dat_sin', ascending=False)
    for id_row, row in sinclim_rea.iterrows() : 
        stormi_loop = row['storm_id']
        storm_landing_date_loop = df_storm_info.loc[df_storm_info.storm_id==stormi_loop]['storm_landing_date']
        valid_dates = stormi_landing_dates_keep[stormi_landing_dates_keep < storm_landing_date_loop.iloc[0]]
        #[date for date in stormi_landing_dates_keep if isinstance(date, datetime.datetime) and date < storm_landing_date_loop]
        closest_earlier_date = max(valid_dates, default=None)
        if closest_earlier_date!=None : 
            new_stormi = df_storm_info.loc[df_storm_info.storm_landing_date == closest_earlier_date]['storm_id']
            sinclim_copy.loc[sinclim_copy['cod_sin'] == row['cod_sin'], 'storm_id'] = new_stormi.iloc[0]
    return sinclim_copy

def gather_claims_storm_closest(sinclim_pd, df_storm_info, nb_min_claims=10):
    """
    Reassign claims linked to storms with fewer than nb_min_claims to the closest storm
    (based on landing date) with more than nb_min_claims.
    
    Args:
        sinclim_pd (pd.DataFrame): DataFrame of claims with storm IDs.
        df_storm_info (pd.DataFrame): DataFrame with storm information, including storm landing dates.
        nb_min_claims (int): Minimum number of claims required to keep a storm.
    
    Returns:
        pd.DataFrame: Updated DataFrame with re-assigned storm IDs for claims.
    """
    df_storm_info = df_storm_info.reset_index(drop=True)
    sinclim_copy = sinclim_pd.copy()
    
    # Group by storm_id to count claims and identify storms with fewer than nb_min_claims
    sinclim_grp = sinclim_copy.groupby('storm_id').size().reset_index(name='n_claims')
    stormi_rea = sinclim_grp.loc[sinclim_grp['n_claims'] < nb_min_claims, 'storm_id'].tolist()
    stormi_keep = sinclim_grp.loc[sinclim_grp['n_claims'] >= nb_min_claims, 'storm_id'].tolist()
    
    # Filter claims to re-assign
    sinclim_rea = sinclim_copy[sinclim_copy['storm_id'].isin(stormi_rea)]
    
    # Filter storm landing dates for storms with enough claims
    stormi_landing_dates_keep = df_storm_info[df_storm_info['storm_id'].isin(stormi_keep)]
    stormi_landing_dates_keep = stormi_landing_dates_keep[['storm_id', 'storm_landing_date']]
    
    # Map storm_landing_date to storm_id for lookup
    storm_landing_dict = stormi_landing_dates_keep.set_index('storm_id')['storm_landing_date'].to_dict()
    
    # Find the closest storm (by date) with enough claims
    def find_closest_storm(row):
        storm_date = df_storm_info.loc[df_storm_info['storm_id'] == row['storm_id'], 'storm_landing_date'].iloc[0]
        # Compute absolute time differences for all valid storms
        closest_storm = min(storm_landing_dict, key=lambda sid: abs(storm_landing_dict[sid] - storm_date))
        return closest_storm
    
    # Apply reassignment logic to the filtered claims
    sinclim_rea = sinclim_rea.copy()
    sinclim_rea.loc[:, 'storm_id'] = sinclim_rea.apply(find_closest_storm, axis=1)
    
    # Update the original DataFrame with reassigned storm IDs
    sinclim_copy.update(sinclim_rea)
    
    return sinclim_copy

def gather_claims_storm_iteration(sinclim_pd, df_storm_info, nb_min_claims, start_min_claims=10, increment=10) :
    """
    Iteratively gather claims based on number associated to individual storms 
    """
    df_storm_info = df_storm_info.reset_index(drop=True)
    sinclim_copy = sinclim_pd.copy()
    
    nb_min_run = start_min_claims 
    while nb_min_run <= nb_min_claims : 
        sinclim_copy = gather_claims_storm_closest(sinclim_copy, df_storm_info, nb_min_run)
        nb_min_run += increment
    return sinclim_copy

## Following function will compute performances of the association 
def performance(sinclim_raw, sinclim_associated, df_storm_info, local_height):
    """
    Evaluate the performance of storm association by checking 
    claim peaks and computing time differences.
    
    Args:
        sinclim_associated (pd.DataFrame): DataFrame of claims with associated storms.
        df_storm_info (pd.DataFrame): DataFrame with storm landing dates.
    
    Returns:
        tuple: (difference in unique storm events, list of minimal date differences)
    """
    # Filter storm info based on associated claims
    storm_ids = sinclim_associated['storm_id'].unique()
    df_storm_info = df_storm_info[df_storm_info['storm_id'].isin(storm_ids)]

    # Aggregate claims by date and identify peaks from the raw claim data 
    dat_sin = sinclim_raw['dat_sin'].value_counts().sort_index()
    dat_sin = dat_sin.asfreq('D').fillna(0)
    peaks, _ = find_peaks(dat_sin.values, height=local_height)
    peak_dates = dat_sin.index[peaks]
    
    # Compute unique storm count difference
    diff_nb_events = sinclim_associated['storm_id'].nunique() - len(peak_dates)#sinclim_associated['dat_sin'].nunique()   
    
    # Compute minimal difference between a claim date and the associated storm
    minimal_dates_diffs = []
    
    # Ensure both columns are datetime
    for storm_id, landing_date in zip(df_storm_info['storm_id'], df_storm_info['storm_landing_date']):
        # Compute minimal difference in days
        min_diff = (peak_dates - landing_date).to_series().abs().min().days
        minimal_dates_diffs.append(min_diff)
    return diff_nb_events, minimal_dates_diffs