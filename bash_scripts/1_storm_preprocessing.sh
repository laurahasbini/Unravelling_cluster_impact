#!/bin/bash
#PBS -N 1_stm_preproc
#PBS -j eo
#PBS -q xlong
#PBS -S /bin/bash

####### Move to path and activate conda environement 
cd /home/users/lhasbini/programs/paper1_Unravelling_wind_impact/tracks_processes
source ~jypeter/.conda3_jyp.sh
conda activate cdatm_py3

###### Convert tracks to TE format 
python tracks_convert_TE.py 
# echo "Done TE convertion"

###### Merge to a single file 
python tracks_merge.py
# echo "Done tracks merge"

###### Filter over tracks impacting France
python tracks_filter_FR.py
# echo "Done filter FR"

###### Create the footprints of each tracks 
python tracks_footprints.py 
python tracks_footprints_varying_radius.py
# echo "Done Footprints"

####### Compute the SSI from footprints 
python tracks_SSI_from_footprint.py 
# echo "Done SSI computed"