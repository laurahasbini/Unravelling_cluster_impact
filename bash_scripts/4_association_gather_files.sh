#!/bin/bash
#PBS -N 4_gather
#PBS -j eo
#PBS -q xlong
#PBS -S /bin/bash

####### Move to path and activate conda environement 
cd /home/users/lhasbini/programs/paper1_Unravelling_wind_impact
source ~jypeter/.conda3_jyp.sh
conda activate cdatm_py3

####### Combine all files 
python -m claim_association.claims_association_combine_files
echo "All files combined"