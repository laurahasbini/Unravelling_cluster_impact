# Open all files and merge 
import pandas as pd
import glob
import os
import time
import datetime as dt
from fct.fct_link_storm_claim import DTYPE_SINCLIM
from fct.paths import *

window        = 'd-3_d+3'
min_claim     = 50
method        = 'wgust'
year_start    = 1997
year_end      = 2023
sinclim_version = 2.2 #2.1 # 2.2
sinclim_peril   = "storm" # "storm_extended" # "storm"

for r, r_varying in zip([900, 1100, 1300, 1300], [False, False, False, True]) : 
    if r_varying : 
        folder_path   = PATH_TEMP + f"final_per_year/r-varying/"
    else : 
        folder_path   = PATH_TEMP + f"final_per_year/r{r}/"

    ## Select files to merge 
    if r_varying:
        file_paths = [
            os.path.join(
                folder_path,
                f"sinclim_v{sinclim_version}_{sinclim_peril}_{window}_unique-{method}_min{min_claim}_priestley_ALL_{y}-{y+1}_r-varying.csv"
            )
            for y in range(year_start, year_end + 1)
        ]
    else:
        file_paths = [
            os.path.join(
                folder_path,
                f"sinclim_v{sinclim_version}_{sinclim_peril}_{window}_unique-{method}_min{min_claim}_priestley_ALL_{y}-{y+1}_r{r}.csv"
            )
            for y in range(year_start, year_end + 1)
        ]

    # Read and merge all CSVs in one go
    dfs = []
    for path_sinclim_storm_unique_wgust in file_paths:
        df = pd.read_csv(path_sinclim_storm_unique_wgust, dtype=DTYPE_SINCLIM)
        dfs.append(df)

    # Concatenate all DataFrames into one
    merged_df = pd.concat(dfs, ignore_index=True)

    if r_varying : 
        fname_out = f"sinclim_v{sinclim_version}_{sinclim_peril}_{window}_unique-{method}_min{min_claim}_priestley_ALL_{year_start}-{year_end+1}_r-varying"+f"_v{dt.datetime.now().strftime('%d%m%y')}"
        merged_df.to_csv(PATH_GENERALI + fname_out+".csv", index=False)
        merged_df.to_parquet(PATH_GENERALI + fname_out+".parquet", index=False)
    else : 
        fname_out = f"sinclim_v{sinclim_version}_{sinclim_peril}_{window}_unique-{method}_min{min_claim}_priestley_ALL_{year_start}-{year_end+1}_r{r}"+f"_v{dt.datetime.now().strftime('%d%m%y')}"
        merged_df.to_csv(PATH_GENERALI + fname_out +".csv", index=False)
        merged_df.to_parquet(PATH_GENERALI + fname_out +".parquet", index=False)

    print(f"All files merged and saved successfully - r{r} varying=[{r_varying}]")
