import datetime as dt
import os
import numpy as np
from shutil import copy


flight = 'UFO8A'

strtDayTm = '20150702_0200'
endDayTm = '20150702_0800'
tDelta = dt.timedelta(minutes=5)


dataDir = '/data/keeling/a/stechma2/radarCompositing/' + flight + '/rawData/'
outDir = '/data/keeling/a/stechma2/radarCompositing/' + flight + '/data_all/'

# Generate target directories for every 5-min period between our start and end times
if not os.path.exists(outDir):
    os.makedirs(outDir)

stDT = dt.datetime.strptime(strtDayTm,'%Y%m%d_%H%M')
endDT = dt.datetime.strptime(endDayTm,'%Y%m%d_%H%M')

curDT = stDT
while curDT <= endDT:
    tgtDir = os.path.join(outDir,dt.datetime.strftime(curDT,'%Y%m%d_%H%M'))
    if not os.path.exists(tgtDir):
        os.mkdir(tgtDir)
    curDT += tDelta

# Create list of full folder paths for each radar we're including
inFolders = []
for (path, dirnames, filenames) in os.walk(dataDir):
    inFolders.extend(os.path.join(path, dirct) for dirct in dirnames)
    
# Create list of full folder paths for each 5-min time we're sorting into
outFolders = []
for (path, dirnames, filenames) in os.walk(outDir):
    outFolders.extend(os.path.join(path, dirct) for dirct in dirnames)


# Create datetime array of our target output times
outTimes = os.listdir(outDir)
outDT = np.asarray([dt.datetime.strptime(outT,'%Y%m%d_%H%M') for outT in outTimes])

# Create array of date_time of all the files in each of our input directories
# Then, convert these to datetimes for comparison with our target times
for dIx in range(0,len(inFolders)):
    print('Now sorting ' + inFolders[dIx][-3:] + '...')
    # Create a list of full file paths within the current radar directory
    inFiles = []
    for (path, dirnames, filenames) in os.walk(inFolders[dIx]):
        inFiles.extend(os.path.join(path, name) for name in filenames)

    inTimes = np.asarray([dTime[4:-4] for dTime in os.listdir(inFolders[dIx])])
    inDT = np.asarray([dt.datetime.strptime(inT,'%Y%m%d_%H%M%S') for inT in inTimes])

    # Check for files in current radar directory that are within 3 min
    # of each of the target times and copies those files to the target directory
    for ix in range(0,len(outDT)):
        tMatch = min(inDT, key=lambda x: abs(x - outDT[ix]))
        if (abs(tMatch - outDT[ix])) >= dt.timedelta(seconds=180):
            continue
        inFileIx = np.squeeze(np.where(inDT == tMatch))
        copy(inFiles[inFileIx],outFolders[ix])

