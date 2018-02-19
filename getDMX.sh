#!/bin/bash

#SBATCH --job-name=get_DMX
#SBATCH --output=DMX.o%j
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=16000
#SBATCH -n 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=stechma2@illinois.edu

HOST='ftp://ftp.ncdc.noaa.gov/pub/has/'
savedir='/data/keeling/a/stechma2/radarCompositing/IOP17/'

dmx="${savedir}rawData/DMX"

if [ ! -d "${savedir}rawData" ]; then
    mkdir "${savedir}rawData"
fi

if [ ! -d "$dmx" ]; then
    mkdir "$dmx"
fi

cd "$dmx"
wget -nv "${HOST}HAS010900627/0001/*.gz"

gunzip *.gz

echo "Download and unzip completed"