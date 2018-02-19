# ## Import modules

import pandas as pd
from netCDF4 import Dataset
import matplotlib
matplotlib.use('AGG')
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import pyart
import numpy as np
import numpy.ma as ma
import os
from glob import glob
import warnings
import datetime as dt
import sys
import matplotlib as mpl

from samuraiAnalysis import samPlt
multXSCrdCalc = samPlt.multXSCrdCalc


# Using this for the runtime warnings generated when determining bad values of lat/long
# Pretty sure it is just because of the NaNs in there, but will want to check into this...
warnings.filterwarnings("ignore",category=RuntimeWarning)


# ## Code options and parameters

flight = '20150709'

# pltLevs = np.arange(0,14) # 0.5-km vertical level increments; 0 = 1 km, 1 = 1.5 km, ...
# pltLevs = [0,1,5,7]
pltLevs = [0]

# Which variable do you want plotted? (Comment out all but one of the following)
pltVar = 'DBZ_qc'
# pltVar = 'VEL_qc'
# pltVar = 'PHV_qc'
# pltVar = 'ZDR_qc'
# pltVar = 'reflectivity'
# pltVar = 'velocity'
# pltVar = 'cross_correlation_ratio'
# pltVar = 'differential_reflectivity'


# Plot NOAA P-3 flight track?
plotFltTrk = False
# Only plot flight track at times of composites?
plotFltStatic = False
# Plot lines indicating sweep locations relative to plane? (Doesn't run if plotFltTrk is False)
plotSwpLocs = True

# Plot ASOS locations? If so, mark a specific location with a star?
plotASOSlocs = False
markStation = False
station = 'KAMA'

# Plot ground-based radar locations?
pltGR = False
pltGRrange = False

# Plot locations of mobile sounding units?
pltMP = False

# Plot locations of fixed assets?
pltFA = False
    
# Want to save output figures?
saveFigs = True
fType = 'png'

# Do you want to zoom in on the plotted map?
# If yes, provide max/min lat/lon values of new domain
mapZoom = False
minLat = 32
maxLat = 45
minLon = -105
maxLon = -95

# Fixed asset locations
faTxt = ['FP1','FP2','FP3','FP4','FP5','FP6','SPOL']
latFA = [36.607, 37.606, 38.958, 40.515, 39.358, 38.144, 38.553]
lonFA = [-97.488, -99.275, -99.574, -98.951, -101.371, -97.439, -99.536]


if flight == '20150617':
    IOP = 'IOP11'
    flF = '20150617I1_AXC.nc'
    
    # Ground-based radar names and locations
    grTxt = ['S1','S2','XP','D6','D7','D8','MX']
    latGR = [40.3214, 40.6473, 40.6674, 40.166, 40.3416, 40.2787, 40.5372]
    lonGR = [-101.009, -101.063, -100.653, -100.621, -100.644, -100.421, -100.35]
    
    # Mobile PISAs
    mpTxt = ['MGS','MISS','MIPS','SPRC','CLMP','NCSU','CSU']
    latMP = [40.5115, 40.5107, 40.6543, 40.5334, 40.2162, 40.5161, 40.216]
    lonMP = [-101.016, -101.016, -100.619, -100.384, -100.743, -100.743, -100.743]
    
elif flight == '20150620':
    IOP = 'UFO4'
    flF = '20150620I1_AXC.nc'
    
    # Ground-based radar names and locations
    grTxt = ['FSD','ABR']
    latGR = [43.583,45.451]
    lonGR = [-96.724,-98.408]
    
    # Mobile PISAs
    mpTxt = []
    latMP = []
    lonMP = []
    
elif flight == '20150701':
    IOP = 'IOP17'
    flF = '20150701I1_AXC.nc'
    
    # Ground-based radar names and locations
    grTxt = ['S1','PX','RX','D6','D7','D8','XP']
    latGR = [40.3788, 40.6099, 40.0903, 40.0964,40.303, 40.3002, 40.5365]
    lonGR = [-95.2464, -95.164, -95.0859, -94.7757, -94.9367, -94.6106, -94.7067]
    
    # Mobile PISAs
    mpTxt = ['MGS','MISS','MIPS','CSU','MG1','MG2']
    latMP = [40.0685, 40.0618, 40.2183, 40.4505, 40.1586, 40.1256]
    lonMP = [-95.6224, -95.6021, -94.551, -95.199, -94.8551, -95.2482]
    
elif flight == '20150702':
    IOP = 'UFO8A'
    flF = '20150702I1_AXC.nc'
    
    # Ground-based radar names and locations
    grTxt = [] # No radars nearby
    latGR = []
    lonGR = []
    
    # Mobile PISAs
    mpTxt = []
    latMP = []
    lonMP = []
    
elif flight == '20150706':
    IOP = 'IOP20'
    flF = '20150706I1_AC.nc'
    
    # Ground-based radar names and locations
    grTxt = ['S1','S2','D6','D7','D8','RX','XP','FSD','ABR']
    latGR = [43.5243,43.2254,43.5547,43.3864,43.1597,43.2427,43.3284,43.583,45.451]
    lonGR = [-98.0047,-98.07,-97.5079,-97.7197,-97.6374,-97.7174,-97.3252,-96.724,-98.408]
    rngGR = [100,100,60,60,90,120,90,315,315]
    
    # Mobile PISAs
    mpTxt = []
    latMP = []
    lonMP = []
    
elif flight == '20150709':
    IOP = 'IOP21'
    flF = '20150709I1_AC.nc'
    
    # Ground-based radar names and locations
    grTxt = ['S1','S2','XP','PX']
    latGR = [35.1807, 35.0467, 35.1822, 34.9633]
    lonGR = [-100.906, -101.239, -101.17, -100.996]
    
    # Mobile PISAs
    mpTxt = []
    latMP = []
    lonMP = []


# Location of radar composite grids
gridDir = ('/data/pecan/a/stechma2/radarCompositing/' + flight + '-' + IOP + '/ReflectivityGrids')
# gridDir = ('/data/pecan/a/stechma2/radarCompositing/' + flight + '-' + IOP + '/ReflectivityGrids_FullHeight')

# Location where plots will be saved
# Create a plots directory within the saveDir if one doesn't already exist
saveDir = gridDir
if saveFigs:
    if not os.path.exists(saveDir + '/plots/'):
        os.makedirs(saveDir + '/plots/')

# Location of aircraft flight-level data file
# flPath = ('/Users/danstechman/GoogleDrive/PECAN-Data/FlightLevelData/' + flF)
flPath = ('/data/pecan/a/stechma2/pecan/flight-level-data/' + flF)

# ASOS location and elevation info
# ASOSinfo = ('/Users/danstechman/GoogleDrive/School/Research/PECAN/Surface/ASOS-sorted.txt')
ASOSinfo = ('/data/pecan/a/stechma2/pecan/ASOS-sorted.txt')

# Assign various variable-specific parameters such as plotting limits and colormaps
if (pltVar == 'reflectivity' or pltVar == 'DBZ_qc'):
    cMin = -8 # Max and min values to plot
    cMax = 72
    cmap = pyart.graph.cm.NWSRef
    norm = None
    saveName = 'refl'
elif (pltVar == 'VEL_qc' or pltVar == 'VEL_qc'):
    cMin = -51
    cMax = 51
    cmap = pyart.graph.cm.Carbone42
    bounds = np.linspace(cMin,cMax,(np.abs(cMin)+np.abs(cMax)+1))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    saveName = 'vel'
elif (pltVar == 'cross_correlation_ratio' or pltVar == 'PHV_qc'):
    cMin = 0
    cMax = 1
    cmap = pyart.graph.cm.RefDiff
    norm = None
    saveName = 'phv'
elif (pltVar == 'differential_reflectivity' or pltVar == 'ZDR_qc'):
    cMin = -7.9
    cMax = 7.9
    cmap = pyart.graph.cm.RefDiff
    norm = None
    saveName = 'zdr'
else:
    sys.exit('You have not entered a valid field to plot')


# ## Helper function to calculate sweep extents

def calcSwpLatLon (lat1,lon1,heading):
    # Sweep unambiguous range (for NOAA TDR - PECAN, 71379 m) converted to radians
    # (relative to spherical Earth, of radius 6372797.6 m)
    beamDistR = 71379./6372797.6
    
    # Calculate headings of both right and left (relative to motion) fore and aft sweeps
    aftLeftHead = heading - (90+20)
    aftRightHead = heading + (90+20)
    foreLeftHead = heading - (90-20)
    foreRightHead = heading + (90-20)
    if aftLeftHead < 0:
        aftLeftHead += 360
    if aftRightHead < 0:
        aftRightHead += 360
    if foreLeftHead < 0:
        foreLeftHead += 360
    if foreRightHead < 0:
        foreRightHead += 360
        
    # Convert headings and coords to radians
    lat1R = np.deg2rad(lat1)
    lon1R = np.deg2rad(lon1)
    aftLeftHeadR = np.deg2rad(aftLeftHead)
    aftRightHeadR = np.deg2rad(aftRightHead)
    foreLeftHeadR = np.deg2rad(foreLeftHead)
    foreRightHeadR = np.deg2rad(foreRightHead)
    
    # Calculate destination coordinates and convert back to degrees
    lat2aftLeftR = np.arcsin(np.sin(lat1R)*np.cos(beamDistR) + np.cos(lat1R)*np.sin(beamDistR)*np.cos(aftLeftHeadR))
    lon2aftLeftR = lon1R + np.arctan2(np.sin(aftLeftHeadR)*np.sin(beamDistR)*np.cos(lat1R), np.cos(beamDistR) - np.sin(lat1R)*np.sin(lat2aftLeftR))
    lat2aftRightR = np.arcsin(np.sin(lat1R)*np.cos(beamDistR) + np.cos(lat1R)*np.sin(beamDistR)*np.cos(aftRightHeadR))
    lon2aftRightR = lon1R + np.arctan2(np.sin(aftRightHeadR)*np.sin(beamDistR)*np.cos(lat1R), np.cos(beamDistR) - np.sin(lat1R)*np.sin(lat2aftRightR))
    lat2foreLeftR = np.arcsin(np.sin(lat1R)*np.cos(beamDistR) + np.cos(lat1R)*np.sin(beamDistR)*np.cos(foreLeftHeadR))
    lon2foreLeftR = lon1R + np.arctan2(np.sin(foreLeftHeadR)*np.sin(beamDistR)*np.cos(lat1R), np.cos(beamDistR) - np.sin(lat1R)*np.sin(lat2foreLeftR))
    lat2foreRightR = np.arcsin(np.sin(lat1R)*np.cos(beamDistR) + np.cos(lat1R)*np.sin(beamDistR)*np.cos(foreRightHeadR))
    lon2foreRightR = lon1R + np.arctan2(np.sin(foreRightHeadR)*np.sin(beamDistR)*np.cos(lat1R), np.cos(beamDistR) - np.sin(lat1R)*np.sin(lat2foreRightR))
    
    latAL = np.rad2deg(lat2aftLeftR)
    lonAL = np.rad2deg(lon2aftLeftR)
    latAR = np.rad2deg(lat2aftRightR)
    lonAR = np.rad2deg(lon2aftRightR)
    latFL = np.rad2deg(lat2foreLeftR)
    lonFL = np.rad2deg(lon2foreLeftR)
    latFR = np.rad2deg(lat2foreRightR)
    lonFR = np.rad2deg(lon2foreRightR)
    
    return latAL,lonAL,latAR,lonAR,latFL,lonFL,latFR,lonFR


# ## Create radar file list and FL Dataset
# Create a list of gridded radar data to loop through and also create a netcdf Dataset object for our flight-level data

# Create list of all pyart gridded netcdf files
radFiles= sorted(glob(gridDir + '/*.nc'))

# Create netcdf Dataset from the specified flight-level file
if plotFltTrk:
    flData = Dataset(flPath,'r')


# ## Read in required FL data
# Read in lat/lon and time data. Also, identify and remove any spikes in the lat/lon data (usually at beginning of flight).

if plotFltTrk:
    if flight == '20150701':
        flLat1 = flData.variables['LatGPS.2'][:]
        flLon1 = flData.variables['LonGPS.2'][:]
    else:
        flLat1 = flData.variables['LatGPS.3'][:]
        flLon1 = flData.variables['LonGPS.3'][:]
    flLat = flLat1.filled() # Fill masked values with fill value (NaN)
    flLon = flLon1.filled()

    # Find spikes in lat/lon and replace with NaN
    latDiff = np.append(0,np.diff(flLat))
    lonDiff = np.append(0,np.diff(flLon))
    badLat = np.squeeze(np.where(np.logical_or(latDiff > 0.1,latDiff < -0.1)))
    badLon = np.squeeze(np.where(np.logical_or(lonDiff > 0.1,lonDiff < -0.1)))
    flLat[badLat] = np.nan
    flLon[badLon] = np.nan

    # Get the flight heading variable (used for plotting sweep locations)
    flHeading = flData.variables['THdgI-GPS.3'][:]
    
    # Get time/date variables and create an array of datetime objects
    flHH1 = flData.variables['HH'][:]
    flMM1 = flData.variables['MM'][:]
    flSS1 = flData.variables['SS'][:]
    flHH = (flHH1.filled()).astype(int)
    flMM = (flMM1.filled()).astype(int)
    flSS = (flSS1.filled()).astype(int)

    flDate = flData.FlightDate
    
    # Account for flights where flight crosses midnight (UTC)
    if flHH[0] > flHH[-1]:
        flEndDate = list(flDate)
        strtD = int(''.join(flEndDate[3:5]))
        endD = str(strtD + 1)
        if int(endD) >= 10:
            flEndDate[3:5] = list(endD)
        else:
            flEndDate[4:5] = list(endD)
        flEndDate = ''.join(flEndDate)

        flStrt = np.array([flDate] * len(np.squeeze(np.where(np.logical_and(flHH >= 18,flHH <= 23)))))
        flEnd = np.array([flEndDate] * len(np.squeeze(np.where(flHH < 14 ))))

        flDateMod = np.append(flStrt,flEnd)

    flDateStr = np.empty(np.shape(flHH),dtype=object)

    if flHH[0] > flHH[-1]:
        for ii in range(0,len(flHH)): 
            flDateStr[ii] = (flDateMod[ii] + '-' + repr(flHH[ii]) + ':' + repr(flMM[ii]) + ':' + repr(flSS[ii]))
    else:
        for ii in range(0,len(flHH)): 
            flDateStr[ii] = (flDate + '-' + repr(flHH[ii]) + ':' + repr(flMM[ii]) + ':' + repr(flSS[ii]))

    flDT = np.asarray([dt.datetime.strptime(fDate,'%m/%d/%Y-%H:%M:%S') for fDate in flDateStr])


# ## Read in ASOS locations
# After loaded, convert lat/lon from deg-min to decimal deg. Additionally, determine the lat/lon of a given station if desired

if plotASOSlocs:
    asosI = pd.read_csv(ASOSinfo,header=0,delim_whitespace=True)

    latD = np.asarray(pd.to_numeric(asosI['LATD'],errors='coerce'))
    latM = np.asarray(pd.to_numeric(asosI['LATM'],errors='coerce'))
    lonD = np.asarray(pd.to_numeric(asosI['LONGD'],errors='coerce'))
    lonM = np.asarray(pd.to_numeric(asosI['LONGM'],errors='coerce'))

    latMd = latM/60.
    lonMd = lonM/60.

    latA = latD+latMd
    lonA = (lonD+lonMd)*-1. # All our stations are in western hemisphere
    
    stationList = np.asarray(asosI['ICAO'].values)
    
    if markStation:
        statIx = np.squeeze(np.where(stationList == station))

        statLat = latA[statIx]
        statLon = lonA[statIx]


# ## Initialize flight plotting variables

# Fill an array with the recorded grid time for each grid file
radDT = np.empty(np.shape(radFiles),dtype=object)
for iz in range(0,len(radFiles)):
    tmpDate = pyart.io.read_grid(radFiles[iz]).time['units'][14:24]
    tmpTime = pyart.io.read_grid(radFiles[iz]).time['units'][25:-1]
    tmpDateStr = (tmpDate + '-' + tmpTime)
    
    radDT[iz] = (dt.datetime.strptime(tmpDateStr,'%Y-%m-%d-%H:%M:%S')).replace(second=0)
       
# Get difference in minutes so that we know how many minutes of flight
# track to plot on each given radar composite
compDiff = np.ones(len(radDT))
if not plotFltStatic:
    for ix in range(1,len(radDT)):
        compDiff[ix-1] = (radDT[ix]-radDT[ix-1]).total_seconds()/60


# ## Plotting

pastTrack = False # We'll set this to true the first time the flight track is in the domain

for ix in range(0,len(radFiles)):

    print('Using grid ' + str(ix+1) + ' of ' + str(len(radFiles)))
    
    # Read in the composite grid and create a GridMapDisplay object
    radarGrid = pyart.io.read_grid(radFiles[ix])
    display = pyart.graph.GridMapDisplay(radarGrid)
    
    # Get height(s) AGL of composite grid(s)
    radAlt = (radarGrid.z['data'][pltLevs])/1000
    
    # Get the lat/lon of each radar used in the composite
    rad88Lats = radarGrid.radar_latitude['data']
    rad88Lons = radarGrid.radar_longitude['data']
    
    
    if plotFltTrk:
        # Find the closest FL data index correlating to the current radar grid time
        domMatch = min(flDT, key=lambda x: abs(x - radDT[ix]))
        flDomIx = np.squeeze(np.where(flDT == domMatch))

        crntFLlat = flLat[flDomIx]
        crntFLlon = flLon[flDomIx]
        crntFLheading = flHeading[flDomIx]
        plotFLlat = flLat[0:flDomIx]
        plotFLlon = flLon[0:flDomIx]
    
    
        # Get the lat/lon data for the grid itself
        # Expand domain by half degree all around - this will help ensure sweep locations
        # are plotted even if plane itself is out of domain
        gridLats = radarGrid.point_latitude['data'][0,:,0]
        gridLons = radarGrid.point_longitude['data'][0,0,:]
        gridLatMin = np.min(gridLats)-0.5
        gridLatMax = np.max(gridLats)+0.5
        gridLonMin = np.min(gridLons)-0.5
        gridLonMax = np.max(gridLons)+0.5
    
        # We want to replot a given composite for every minute we have flight (unless plotFltStatic is True)
        # data until the next composite
        # We'll only make a plot for every minute when the plane is actually in the air and within 0.5 deg of the domain
        if ((flDT[0]<= radDT[ix] <= flDT[-1]) & (gridLatMin <= crntFLlat <= gridLatMax) & (gridLonMin <= crntFLlon <= gridLonMax)):
            inDomain = True
            inLoop = int(compDiff[ix])
            if inLoop > 9:
                inLoop = 1;
                print('Difference between grids exceeded 9 minutes - skipping 1-min flt track')
            pastTrack = True
        else:
            inDomain = False
            inLoop = 1
    else:
        inLoop = 1
        inDomain = False
        
    
    for iz in range(0,inLoop):
        pltT = radDT[ix] + dt.timedelta(minutes=iz)
        print('\tNow plotting ' + str(pltT) + '...')
        
        
        for iL in pltLevs:
            print('\t\tLevel = {:.1f} km'.format(radAlt[iL]))
            # Create figure object
            fig = plt.figure(figsize=[13,10])

            # If the zoom option is enabled, use the min/max lat/lon to define the basemap boundaries
            if mapZoom:
                    display.plot_basemap(resolution='i',auto_range=False,
                                         min_lon=minLon,max_lon=maxLon,
                                         min_lat=minLat,max_lat=maxLat,
                                         stateBndLW=1.0,gridDash=[1,3])
            else:
                display.plot_basemap(lat_lines=np.arange(30, 50, 1),resolution='i',stateBndLW=1.0,gridDash=[1,3])
        
        
            # Plot the gridded data
            display.plot_grid(pltVar,level=pltLevs[iL],vmin=cMin,vmax=cMax,
                          title=(repr(radAlt[iL]) + ' km AGL ' + pltVar + '\n' + 
                                 dt.datetime.strftime(pltT,'%Y-%m-%d %H:%M') + ' UTC'),
                          cmap=cmap,norm=norm,fig=fig)
            ax1 = plt.gca()
            ax1.set_title(str(ax1.get_title()),fontsize=20)



            # If our radar grid time is within bounds of the flight, plot the flight track
            # This will automatically be False if plot_FltTrk is False
            if pastTrack:
            
                domMatch = min(flDT, key=lambda x: abs(x - pltT))
                flDomIx = np.squeeze(np.where(flDT == domMatch))

                crntFLlat = flLat[flDomIx]
                crntFLlon = flLon[flDomIx]
                crntFLheading = flHeading[flDomIx]
                plotFLlat = flLat[0:flDomIx]
                plotFLlon = flLon[0:flDomIx]

                # Plot flight track from beginning to time neareast current composite
                x,y = display.basemap(plotFLlon, plotFLlat)
                display.basemap.plot(x,y,lw=2.25,color = 'k',zorder=15)
                display.basemap.plot(x,y,lw=.75,color = 'w',zorder=15)
            
                # If the plane is within 0.5 deg box of domain, plot at least the plane
                # location, and optionally, the sweep locations
                if inDomain:
                    if plotSwpLocs:
                        latAL,lonAL,latAR,lonAR,latFL,lonFL,latFR,lonFR = calcSwpLatLon(plotFLlat[-1],plotFLlon[-1],crntFLheading)
                        xALend,yALend = display.basemap(lonAL, latAL)
                        xARend,yARend = display.basemap(lonAR, latAR)
                        xFLend,yFLend = display.basemap(lonFL, latFL)
                        xFRend,yFRend = display.basemap(lonFR, latFR)

                        # Create x/y coordinate arrays with the start/end values for each sweep
                        xAL = [x[-1],xALend]
                        xAR = [x[-1],xARend]
                        xFL = [x[-1],xFLend]
                        xFR = [x[-1],xFRend]
                        yAL = [y[-1],yALend]
                        yAR = [y[-1],yARend]
                        yFL = [y[-1],yFLend]
                        yFR = [y[-1],yFRend]

                        # Plot lines indicating sweep locations
                        display.basemap.plot(xAL,yAL,lw=3,color='w',zorder=15)
                        display.basemap.plot(xAL,yAL,lw=1.75,color='MediumBlue',zorder=15)
                        display.basemap.plot(xAR,yAR,lw=3,color='w',zorder=15)
                        display.basemap.plot(xAR,yAR,lw=1.75,color='MediumBlue',zorder=15)
                        display.basemap.plot(xFL,yFL,lw=3,color='w',zorder=15)
                        display.basemap.plot(xFL,yFL,lw=1.75,color='DarkRed',zorder=15)
                        display.basemap.plot(xFR,yFR,lw=3,color='w',zorder=15)
                        display.basemap.plot(xFR,yFR,lw=1.75,color='DarkRed',zorder=15)
                    
                    # Plot marker at current location of plane
                    display.basemap.plot(x[-1],y[-1], 'wo', markersize = 12,zorder=15)
                    display.basemap.plot(x[-1],y[-1], 'ko', markersize = 8,zorder=15)

            # Plot ASOS locations and names
            if plotASOSlocs:
                lonAT,latAT = display.basemap(lonA,latA)
                display.basemap.scatter(lonAT,latAT,marker='o',color='k',s=50)

                if markStation:
                    sLonAT,sLatAT = display.basemap(statLon,statLat)
                    display.basemap.scatter(sLonAT,sLatAT,marker='*',color='w',s=800,zorder=7)
                    display.basemap.scatter(sLonAT,sLatAT,marker='*',color='b',s=500,zorder=8)

                for label, a, b in zip(stationList, lonAT, latAT):
                    plt.annotate(label, xy = (a, b), xytext = (4, 6),zorder=10,
                                 textcoords = 'offset points',fontsize=11,
                                 bbox=dict(boxstyle="round", fc="w",alpha=0.4,pad=0.01))

            # Plot location of each 88D used in composite, along with the locations of any ground assets
            if pltGR:
                rad88LonsT,rad88LatsT = display.basemap(rad88Lons,rad88Lats)
                display.basemap.scatter(rad88LonsT,rad88LatsT,marker='D',color='w',s=200,zorder=7)
                display.basemap.scatter(rad88LonsT,rad88LatsT,marker='D',color='k',s=100,zorder=8)

                lonGRt,latGRt = display.basemap(lonGR,latGR)
                display.basemap.scatter(lonGRt,latGRt,marker='d',color='w',s=200,zorder=7)
                display.basemap.scatter(lonGRt,latGRt,marker='d',color='k',s=100,zorder=8)

                for labelGR, c, d in zip(grTxt, lonGRt, latGRt):
                    plt.annotate(labelGR, xy = (c, d), xytext = (-8, 5),zorder=10,
                                 textcoords = 'offset points',fontsize=13,color='w',
                                 bbox=dict(boxstyle="round", fc="b",alpha=0.6,pad=0.01))
                             
                if pltGRrange:
                    for iGR in range(0,len(grTxt)):
                        latCirc = []
                        lonCirc = []
                        for az in range(0,364):
                            tempLat,tempLon = multXSCrdCalc(latGR[iGR],lonGR[iGR],rngGR[iGR],az)
                            latCirc.append(tempLat)
                            lonCirc.append(tempLon)
                        lonCircT,latCircT = display.basemap(lonCirc,latCirc)
                        if iGR%2 == 0:
                            display.basemap.plot(lonCircT,latCircT,linestyle='-',linewidth=3,color='w')
                            display.basemap.plot(lonCircT,latCircT,label=grTxt[iGR],linestyle='-',linewidth=2)
                        else:
                            display.basemap.plot(lonCircT,latCircT,linestyle='-',linewidth=3,color='w')
                            display.basemap.plot(lonCircT,latCircT,label=grTxt[iGR],linestyle='--',linewidth=2)

                    ax1.legend(fontsize=14,loc='upper right')
                
            # Plot locations of mobile sounding units
            if pltMP:
                lonMPt,latMPt = display.basemap(lonMP,latMP)
                display.basemap.scatter(lonMPt,latMPt,marker='*',color='w',s=200,zorder=7)
                display.basemap.scatter(lonMPt,latMPt,marker='*',color='k',s=100,zorder=8)

                for labelMP, e, f in zip(mpTxt, lonMPt, latMPt):
                    plt.annotate(labelMP, xy = (e, f), xytext = (-8, 5),zorder=10,
                                 textcoords = 'offset points',fontsize=13,color='k',
                                 bbox=dict(boxstyle="round", fc="r",alpha=0.6,pad=0.01))

            # Plot locations of mobile sounding units
            if pltFA:
                lonFAt,latFAt = display.basemap(lonFA,latFA)
                display.basemap.scatter(lonFAt,latFAt,marker='v',color='w',s=200,zorder=7)
                display.basemap.scatter(lonFAt,latFAt,marker='v',color='k',s=100,zorder=8)

                for labelFA, g, h in zip(faTxt, lonFAt, latFAt):
                    plt.annotate(labelFA, xy = (g, h), xytext = (-8, 5),zorder=10,
                                 textcoords = 'offset points',fontsize=13,color='w',
                                 bbox=dict(boxstyle="round", fc="k",alpha=0.6,pad=0.01))
                
            plt.tight_layout(pad=1.5)

            if saveFigs:
                if plotASOSlocs:
                    asosStr = '-asos'
                else:
                    asosStr = ''
                
                if pltGRrange:
                    grRngStr = '-radRng'
                else:
                    grRngStr = ''
                    
                if plotFltTrk:
                    fltTrkStr = '-fltTrk'
                else:
                    fltTrkStr = ''
                    
                if pltGR:
                    grStr = '-GR'
                else:
                    grStr = ''
                    
                if pltMP:
                    mpStr = '-MP'
                else:
                    mpStr = ''
                    
                    
                fig.savefig('{}/plots/{}_{:.1f}km-{}{}{}{}{}{}_{}.{}'.format(saveDir,dt.datetime.strftime(pltT,'%Y%m%d'),radAlt[iL],
                            saveName,fltTrkStr,asosStr,grStr,mpStr,grRngStr,dt.datetime.strftime(pltT,'%H%M'),fType),bbox_inches='tight',dpi=100)
                # if pltGRrange:
#                     # fig.savefig(saveDir + '/plots/' + dt.datetime.strftime(pltT,'%Y%m%d_') + repr(radAlt[iL]) + 'km-'
# #                                 + saveName + '-fltTrk-radRng_' + dt.datetime.strftime(pltT,'%H%M.') + fType,bbox_inches='tight',dpi=100)
#                     fig.savefig('{}/plots/{}_{:.1f}km-{}-fltTrk-radRng_{}.{}'.format(saveDir,dt.datetime.strftime(pltT,'%Y%m%d'),radAlt[iL],
#                                 saveName,dt.datetime.strftime(pltT,'%H%M'),fType),bbox_inches='tight',dpi=100)
#                 else:
#                     # fig.savefig(saveDir + '/plots/' + dt.datetime.strftime(pltT,'%Y%m%d_') + repr(radAlt[iL]) + 'km-'
# #                                 + saveName + '-fltTrk-asos_' + dt.datetime.strftime(pltT,'%H%M.') + fType,bbox_inches='tight',dpi=100)
#                     fig.savefig('{}/plots/{}_{:.1f}km-{}-fltTrk-asos_{}.{}'.format(saveDir,dt.datetime.strftime(pltT,'%Y%m%d'),radAlt[iL],
#                                 saveName,dt.datetime.strftime(pltT,'%H%M'),fType),bbox_inches='tight',dpi=100)

            plt.close('all')