#!/bin/bash
#PBS -N 2_assoc_per_year_subset
#PBS -j eo
#PBS -q xlongp
#PBS -S /bin/bash
#PBS -l select=1:ncpus=10

cd /home/users/lhasbini/programs/paper1_Unravelling_wind_impact
source ~jypeter/.conda3_jyp.sh
conda activate cdatm_py3


# Loop over all radius 
for r in 900 1100 1300; do
# for r in 1300; do
    # Loop over all years
    for year in $(seq 1997 2023); do
        python -m claim_association.claims_association_storms_per_year "$year" "$r"
    done
done
