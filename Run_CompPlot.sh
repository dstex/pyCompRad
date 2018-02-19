#!/bin/bash

#SBATCH --job-name=plotComp_IOP20
#SBATCH --output=plotComp_IOP20.o%j
#SBATCH --time=24:00:00
#SBATCH --mem=32000
#SBATCH -n 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=stechma2@illinois.edu

PYFILE="/data/keeling/a/stechma2/autoRadarComp/plt_Composites.py"

T1=$(date +%s)

stdbuf -oL python "$PYFILE"

T2=$(date +%s)
diffsec="$(expr $T2 - $T1)"
echo | awk -v D=$diffsec '{printf "Elapsed time: %02d:%02d:%02d\n",D/(60*60),D%(60*60)/60,D%60}'