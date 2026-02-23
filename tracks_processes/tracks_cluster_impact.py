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
period = "1979-2024WIN"
cat = "FR_ALL_24h"

#Tracks lasting more than 24h
df_info_storm                       = pd.read_csv(PATH_TRACKS+f"tracks_{cat}_{period}_info.csv", encoding='utf-8')
df_info_storm['storm_landing_date'] = pd.to_datetime(df_info_storm['storm_landing_date'])
df_storm                            = pd.read_csv(PATH_TRACKS+f"tracks_{cat}_{period}.csv", encoding='utf-8')

# Import de the Europe shapefile 
FRA_geo_shp = gpd.read_file(os.path.join(PATH_SHP_COUNTRIES, "world-administrative-boundaries_FRA_level0.shp"))

########## IMPACT DATA 
window        = "d-3_d+3"
min_claims_range = [50]
method        = 'wgust'
year_start    = 1997
year_end      = 2024
version_date  = "v050126"
sinclim_version = 2.2 #2.1 # 2.2
sinclim_peril   = "storm_extended" # "storm_extended" # "storm"

########## CLUSTER 
cluster_window = [72, 96, 120]
radius = [None] #[700, None]
# r_sinclim = 900 # 1100 # 1300 
# r_varying = False #True 

for r_sinclim, r_varying in zip([1300], [False]) : 
    for min_claim in min_claims_range : 
        if r_varying : 
            fname_in = f"sinclim_v{sinclim_version}_{sinclim_peril}_{window}_unique-{method}_min{min_claim}_priestley_ALL_{year_start}-{year_end}_r-varying_{version_date}" 
#             sinclim       = pd.read_csv(PATH_GENERALI+fname_in+".csv", low_memory=False, dtype={'cod_sin': str, "cod_pol" : str})
            sinclim       = pd.read_parquet(PATH_GENERALI+fname_in+".parquet")
        else : 
            fname_in = f"sinclim_v{sinclim_version}_{sinclim_peril}_{window}_unique-{method}_min{min_claim}_priestley_ALL_{year_start}-{year_end}_r{r_sinclim}_{version_date}"
#             sinclim       = pd.read_csv(PATH_GENERALI+fname_in+".csv", low_memory=False, dtype={'cod_sin': str, "cod_pol" : str})
            sinclim       = pd.read_parquet(PATH_GENERALI+fname_in+".parquet")
            
        stromi_impact = sinclim.storm_id.unique()

#         df_info_storm['FR_ALL_24h'] = df_info_storm['FR_ALL_24h_original'].loc[df_info_storm['FR_ALL_24h_original'].storm_id.isin(stromi_impact)]
#         df_storm['FR_ALL_24h']      = df_storm['FR_ALL_24h_original'].loc[df_storm['FR_ALL_24h_original'].storm_id.isin(stromi_impact)]
        df_info_storm_impact = df_info_storm.loc[df_info_storm.storm_id.isin(stromi_impact)]
        df_storm_impact             = df_storm.loc[df_storm.storm_id.isin(stromi_impact)]
        for nb_hours in cluster_window : 
            for r in radius : 
                r_part = f"_r{r}" if r is not None else ""
                
                df_cluster = {}
                df_info_cluster = {}

                #### RUN FOR STORM IN MULTIPLE CLUSTERS 
                df_info_cluster = storm_eu_cluster.assign_clusters_allow_multiple(df_info_storm_impact, df_storm_impact, r=r, nb_hours_diff=nb_hours)
                df_info_mult_cluster = storm_eu_cluster.filter_clusters_explode(df_info_cluster)
                df_mult_cluster = df_storm.loc[df_storm.storm_id.isin(df_info_mult_cluster.storm_id.unique())]

                if r_varying:
                    fname_storm_out = (
                        f"tracks_{cat}_impact_v{sinclim_version}_{sinclim_peril}_"
                        f"{window}-unique-{method}_min{min_claim}_r-varying_"
                        f"{version_date}_clust-mult-2storms-{nb_hours}h"
                        f"{r_part}_{period}"
                    )
                else:
                    fname_storm_out = (
                        f"tracks_{cat}_impact_v{sinclim_version}_{sinclim_peril}_"
                        f"{window}-unique-{method}_min{min_claim}_r{r_sinclim}_"
                        f"{version_date}_clust-mult-2storms-{nb_hours}h"
                        f"{r_part}_{period}"
                    )

                df_mult_cluster.to_csv(PATH_TRACKS + fname_storm_out + ".csv",
                                       encoding="utf-8", index=False)
                df_info_mult_cluster.to_csv(PATH_TRACKS + fname_storm_out + "_info.csv",
                                            encoding="utf-8", index=False)
#                         df_mult_cluster.to_csv(PATH_TRACKS+f"tracks_{cat}_impact_{window}unique-{method}_min{min_claim}_r{r_sinclim}_clust-mult-2storms-{nb_hours}h_r{r}_{period}.csv", 
#                                                encoding='utf-8', index=False) 
#                         df_info_mult_cluster.to_csv(PATH_TRACKS+f"tracks_{cat}_impact_{window}unique-{method}_min{min_claim}_r{r_sinclim}_clust-mult-2storms-{nb_hours}h_r{r}_{period}_info.csv", 
#                                                     encoding='utf-8', index=False)


