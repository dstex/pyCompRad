#!/bin/bash

#SBATCH --job-name=sort88D
#SBATCH --output=sort88D.o%j
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH -n 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=stechma2@illinois.edu

PYFILE="/data/keeling/a/stechma2/autoRadarComp/sort88Dfiles.py"

T1=$(date +%s)

stdbuf -oL python "$PYFILE"

T2=$(date +%s)
diffsec="$(expr $T2 - $T1)"
echo | awk -v D=$diffsec '{printf "Elapsed time: %02d:%02d:%02d\n",D/(60*60),D%(60*60)/60,D%60}'