import math
import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import xarray as xr 

from fct.paths import *

def open_sinclim_associated(path_generali, window, min_claim, method, period, r) : 
    sinclim       = pd.read_csv(path_generali+"sinclim_v2.1_storm_"+window+"_unique-"+method+"_min"+str(min_claim)+"_priestley_ALL_"+period+"_r"+str(r)+".csv", low_memory=False)
    sinclim['dat_sin'] = pd.to_datetime(sinclim['dat_sin'])
    
    #Remove grave sinisters 
    sinclim = sinclim.loc[sinclim.num_chg_brut < 150000]
    
    #Remove negative losses
    sinclim = sinclim.loc[sinclim.num_chg_brut > 0]
    
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
    
    return sinclim 