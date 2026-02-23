import pandas as pd
import glob
import os
import datetime as dt
from fct.paths import *

# Path to the main folder
base_path = os.path.join(PATH_GENERALI, "sinclim_v101225")

# Find all parquet files recursively
file_list = glob.glob(os.path.join(base_path, "survenance=*/part-0.parquet"))

print(f"Found {len(file_list)} files")

# Read and concatenate all parquet files
df = pd.concat([pd.read_parquet(f) for f in file_list], ignore_index=True)

# Rename columns and anom 
columns_rename = {"POINT_X" : "num_lon", 
                  "POINT_Y" : "num_lat",
                  "geocoding_quality" : "lib_geo", 
                  "Postal" : "cod_iso"}
columns_keep = ['cod_sin', 'cod_pol', 'dat_sin', 'dat_dec', 'dat_clo', 'cod_cie',
       'cod_ent', 'cod_res', 'lib_res', 'lib_eta', 'lib_lob', 'lib_lob2',
       'lib_per', 'num_chg_brut', 'cod_csd', 'cod_reg', 'cod_bil', 'cod_nat',
       'cod_cbo', 'cod_iso', 'lib_geo', 'num_lat', 'num_lon', 'dat_maj']

df_final = df.rename(columns_rename, axis=1)
df_final= df_final[columns_keep]

# Save the combined file
output_path = os.path.join(PATH_GENERALI, f"sinclim_v2.2_anom_v{dt.datetime.now().strftime('%d%m%y')}.parquet")
df_final.to_parquet(output_path, index=False)

print(f"Saved combined parquet to {output_path}")