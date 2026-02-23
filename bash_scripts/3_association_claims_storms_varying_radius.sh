#!/bin/bash
#PBS -N 3_association_varyng_radius
#PBS -j eo
#PBS -q xlongp
#PBS -S /bin/bash
#PBS -l select=1:ncpus=13

cd /home/users/lhasbini/programs/paper1_Unravelling_wind_impact
source ~jypeter/.conda3_jyp.sh
conda activate cdatm_py3

# Loop over all years
for year in $(seq 1997 2023); do
    python -m claim_association.claims_association_storms_varying_radius_per_year "$year"
done
