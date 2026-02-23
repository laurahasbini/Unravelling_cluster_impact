import math
import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import xarray as xr 
from scipy.interpolate import griddata
from fct.fct_link_storm_claim import DTYPE_SINCLIM
from fct.paths import *

def open_sinclim_associated(path_generali, sinclim_v="v2.2", sinclim_peril="storm", sinclim_date=None, window="d-3_d+3", min_claim=50, method="wgust", period="1979-2024WIN", r=1300, r_varying=False) : 
    sinclim_date_part = f"_{sinclim_date}" if sinclim_date is not None else ""
    r_part = f"r-varying" if r_varying else f"r{r}"
    sinclim       = pd.read_csv(path_generali+f"sinclim_{sinclim_v}_{sinclim_peril}_{window}_unique-{method}_min{min_claim}_priestley_ALL_{period}_{r_part}{sinclim_date_part}.csv",
                                dtype=DTYPE_SINCLIM, 
                                low_memory=False)
    sinclim['dat_sin'] = pd.to_datetime(sinclim['dat_sin'])
    
    #Remove grave sinisters 
    sinclim = sinclim.loc[sinclim.num_chg_brut < 150000]
    
    #Remove negative losses
    sinclim = sinclim.loc[sinclim.num_chg_brut > 0]
    
    #Verify that the year is in sinclim columns 
    if "year" not in sinclim.columns :
        sinclim['year'] = sinclim['dat_sin'].dt.year
    
    #Remove the inflation
    insee_coeff      = pd.read_csv(path_generali+'insee_constant_euros.csv', sep=";")
    columns_year = [str(y) for y in range(2024, 1996, -1)]
    inflation_data = {
        'year': columns_year,
        'coef_inflation': insee_coeff.loc[0,columns_year].tolist()
    }
    inflation_data = pd.DataFrame(inflation_data)
    inflation_data.coef_inflation = inflation_data.coef_inflation.astype(float).fillna(0.0)
    inflation_data.year = inflation_data.year.astype(int).fillna(0.0)

    sinclim = sinclim.rename(columns={"survenance" : "year"})
    sinclim = pd.merge(sinclim, inflation_data, on='year', how='left')
    sinclim['num_chg_brut_cst'] = sinclim['num_chg_brut'] * sinclim['coef_inflation']/100
    
    # Remove potential duplicates 
    sinclim = sinclim.drop_duplicates()
    return sinclim 

def add_lat(row, grid_lat, latitude_name) : 
    try : 
        return(grid_lat[np.abs(grid_lat - row[latitude_name]).argmin()])
    except (KeyError, ValueError, AttributeError):
        return np.nan

def add_lon(row, grid_lon, longitude_name) : 
    try : 
        return(grid_lon[np.abs(grid_lon - row[longitude_name]).argmin()])
    except (KeyError, ValueError, AttributeError):
        return np.nan

def grid_df(df, grid_lat, grid_lon, latitude_name, longitude_name): 
    grid_lon_mesh, grid_lat_mesh = np.meshgrid(grid_lon, grid_lat)

    #Interpolate 
    interpolated_lat = griddata(
        points=(df[longitude_name], df[latitude_name]),
        values=df[latitude_name],
        xi=(grid_lon_mesh, grid_lat_mesh),
        method='linear'
    )
    interpolated_lon = griddata(
        points=(df[longitude_name], df[latitude_name]),
        values=df[longitude_name],
        xi=(grid_lon_mesh, grid_lat_mesh),
        method='linear'
    )
    interpolated_lat_flat = interpolated_lat.flatten()
    interpolated_lon_flat = interpolated_lon.flatten()

    df['latitude']  = df.apply(lambda row: add_lat(row, grid_lat, latitude_name), axis=1)
    df['longitude'] = df.apply(lambda row: add_lon(row, grid_lon, longitude_name), axis=1)
    
    return df 

def add_wind_quantiles_era5_xr(sinclim_xr) : 
    for q in [90, 95, 98] : 
        wind_th = xr.open_mfdataset(PATH_WGUST_QUANTILE+"windgust_percentile-q"+str(q)+"_1950_2022_winter.nc", 
                                    chunks={'time':-1, 'latitude':'auto', 'longitude':'auto'})
        wind_th = wind_th.isel(bnds=0).isel(time=0, drop=True)
        wind_th['fg10'] = xr.where(wind_th['fg10'] < 9, 9, wind_th['fg10'])    
        
        
        interp_th = wind_th['fg10'].interp(
            latitude=sinclim_xr['latitude'],
            longitude=sinclim_xr['longitude'],
            method='nearest'
        )
        sinclim_xr[f'wgust_q{q}'] = interp_th
   
        #Wind ratio 
        sinclim_xr[f'wind_ratio_q{q}'] = sinclim_xr["wgust"]/sinclim_xr[f'wgust_q{q}']
    return sinclim_xr
