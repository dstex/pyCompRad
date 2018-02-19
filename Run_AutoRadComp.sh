#!/bin/bash

#SBATCH --job-name=compRad_IOP20
#SBATCH --output=compRad_IOP20.o%j
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH -n 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=stechma2@illinois.edu

WKDIR=`pwd`
PYFILE="/data/keeling/a/stechma2/autoRadarComp/AutoRadarComp.py"

T1=$(date +%s)

for D in data/*; do
	TIME=${D:5}
	printf "%s\tWorking in directory: %s\n" "$(date)" "$D"
	if [ "$TIME" -ge 0000 -a "$TIME" -le 0855 ]; then
		GLAT=43.3864
		GLON=-97.7197
	fi
	stdbuf -oL python "$PYFILE" -t $D -w $WKDIR -x $GLON -y $GLAT
done

T2=$(date +%s)
diffsec="$(expr $T2 - $T1)"
echo | awk -v D=$diffsec '{printf "Elapsed time: %02d:%02d:%02d\n",D/(60*60),D%(60*60)/60,D%60}'