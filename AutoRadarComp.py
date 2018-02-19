## Written by Dan Stechman, with original code base by Jessie Choate ##

import pyart
import getopt
import sys
import time
import os
import numpy as np
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from csu_radartools import (csu_misc, csu_kdp)


cleanSweeps = True
debugClean = False



# Available fields in level 2 data: 'reflectivity', 'velocity', 'spectrum_width', 
#                                   'differential_phase', 'differential_reflectivity', 
#                                   'cross_correlation_ratio'

# Which fields to exclude from raw 88D files during import?
excluded_fields = []

# Which fields to include in output grids? 
if debugClean:
    included_fields = ['reflectivity', 'velocity', 'differential_reflectivity', 'cross_correlation_ratio', 'DBZ_insect', 'DBZ_SDdp', 'DBZ_preDespeck', 
    'DBZ_qc','VEL_qc','ZDR_qc','PHV_qc']
else:
    included_fields = ['DBZ_qc','VEL_qc','ZDR_qc','PHV_qc','velocity']


# Option to process only part of the radar volume
thinVol = False
keep_nsweeps = 4 # Keeps lowest 4 sweeps of radar volume - speeds up computation



# Read in arguments from command line or script
#     gridTime: HHMM of individual composite (times based on directory structure of data)
#    wrkDir: working directory
#    gridLon/gridLat: Longitude/Latitude of grid origin (can change based on gridTime)
def autoRadCmp(argv):
    try:
        opts, args = getopt.getopt(argv,'t:w:x:y:',['gridTime=', 'wrkDir=', 'gridLon=', 'gridLat='])
    except getopt.GetoptError:
        sys.exit('Something went wrong with directory assignment.') 
    for opt, arg in opts:
        if opt in ('-t','--gridTime'):
            gridTime = str(arg)
        if opt in ('-w','--wrkDir'):
            wrkDir = str(arg)
        if opt in ('-x','--gridLon'):
            gridLon = float(arg)
        if opt in ('-y','--gridLat'):
            gridLat = float(arg)


    # Extract HHMM time from file name
    truGridT = gridTime.replace('data/', '')
    print ('\nWorking on time {}'.format(truGridT))
    print ('\nUsing grid origin of ({}, {})'.format(gridLat,gridLon))

    wrkPath = wrkDir + '/' + gridTime


    # List all files which will be meshed together for current grid time
    RadFiles = []
    print ('\nThe following radar files will be meshed:')
    for f in listdir(wrkPath):
        print (f)
        if isfile(join(wrkPath,f)) and ('V06' in f):
            RadFiles.append(f)
    
    
    # Read in data from every radar file for the given grid time and optionally
    # exclude variables and/or thin the radar volume
    if cleanSweeps:
        print ('\nNow cleaning up radar data {}'.format(time.asctime()))
    radData = {}
    for x in range(0, len(RadFiles)):
        dataFName = wrkPath + '/' + RadFiles[x]
        tmpRadName = 'RADAR_{0}'.format(x)
        radData[tmpRadName] = pyart.io.read(dataFName, exclude_fields=excluded_fields)
        if thinVol:
             radData[tmpRadName] = (radData[tmpRadName]).extract_sweeps(range(keep_nsweeps))
        if cleanSweeps:
             # Value used as a bad data flag
             bdf = -32768
             
             # Retrieve unmasked data from radar object
             zdrN = radData[tmpRadName].fields['differential_reflectivity']['data'].filled(fill_value=bdf)
             dbzN = radData[tmpRadName].fields['reflectivity']['data'].filled(fill_value=bdf)
             dpN = radData[tmpRadName].fields['differential_phase']['data'].filled(fill_value=bdf)
             velN = radData[tmpRadName].fields['velocity']['data'].filled(fill_value=bdf)
             phvN = radData[tmpRadName].fields['cross_correlation_ratio']['data'].filled(fill_value=bdf)
             
             # Create 2D range and azimuth variables for use in the KDP calculation
             rng2d, az2d = np.meshgrid(radData[tmpRadName].range['data'], radData[tmpRadName].azimuth['data'])
             
             # Calculate specific differential phase (kdpN), filtered differential phase (fdpN)
             # (using finite impulse response [FIR] filter), and the standard deviation of
             # differential phase (SDdpN)
             kdpN, fdpN, SDdpN = csu_kdp.calc_kdp_bringi(dp=dpN, dz=dbzN, rng=rng2d/1000.0)
             
             
             # Determine various masks on the data
             insect_mask = csu_misc.insect_filter(dbzN, zdrN)
             SDdp_mask = csu_misc.differential_phase_filter(SDdpN, thresh_sdp=13)
             phase_insect_mask = np.logical_or(insect_mask, SDdp_mask)

             # Create new reflectivity array masked only by insects
             dbz_insect = 1.0 * dbzN
             dbz_insect[insect_mask] = bdf
                       
             # Create new reflectivity array masked only by the SD of diff. phase
             dbz_SDdp = 1.0 * dbzN
             dbz_SDdp[SDdp_mask] = bdf
                 
             # Create new arrays masked by both insects and the SD of diff. phase
             dbz_qc = 1.0 * dbzN
             zdr_qc = 1.0 * zdrN
             vel_qc = 1.0 * velN
             phv_qc = 1.0 * phvN
             dbz_qc[phase_insect_mask] = bdf
             zdr_qc[phase_insect_mask] = bdf
             vel_qc[phase_insect_mask] = bdf
             phv_qc[phase_insect_mask] = bdf
                   
             # Now create despeckling mask based on reflectivity and apply to other fields
             dbz_pre_despeck = 1.0 * dbz_qc
             mask_despeck = csu_misc.despeckle(dbz_qc, ngates=4)
             dbz_qc[mask_despeck] = bdf
             zdr_qc[mask_despeck] = bdf
             vel_qc[mask_despeck] = bdf
             phv_qc[mask_despeck] = bdf
                   
                   
             # Add modified variables to radar object, including a number of
             # intermediary reflectivity variables to help debug if necessary
                                   
             if debugClean:
                 radData[tmpRadName] = add_field_to_radar_object(fdpN, radData[tmpRadName], field_name='FDP', units='deg', 
                                       long_name='Filtered Differential Phase',
                                       standard_name='Filtered Differential Phase', 
                                       dz_field='reflectivity')
                 radData[tmpRadName] = add_field_to_radar_object(SDdpN, radData[tmpRadName], field_name='SDP', units='deg', 
                                       long_name='Standard Deviation of Differential Phase',
                                       standard_name='Standard Deviation of Differential Phase', 
                                       dz_field='reflectivity')
                 radData[tmpRadName] = add_field_to_radar_object(dbz_insect, radData[tmpRadName], field_name='DBZ_insect', units='dBZ', 
                                       long_name='Reflectivity (Insect Filtered)',
                                       standard_name='Reflectivity (Insect Filtered)', 
                                       dz_field='reflectivity')
                 radData[tmpRadName] = add_field_to_radar_object(dbz_SDdp, radData[tmpRadName], field_name='DBZ_SDdp', units='dBZ', 
                                       long_name='Reflectivity (SD-DP Filtered)',
                                       standard_name='Reflectivity (SD-DP Filtered)', 
                                       dz_field='reflectivity')
                 radData[tmpRadName] = add_field_to_radar_object(dbz_pre_despeck, radData[tmpRadName], field_name='DBZ_preDespeck', units='dBZ', 
                                       long_name='Reflectivity (No Despeckling)',
                                       standard_name='Reflectivity (No Despeckling)', 
                                       dz_field='reflectivity')
             
             radData[tmpRadName] = add_field_to_radar_object(dbz_qc, radData[tmpRadName], field_name='DBZ_qc', units='dBZ', 
                                   long_name='Reflectivity (Insect/SD-DP Filtered)',
                                   standard_name='Reflectivity (Insect/SD-DP Filtered)', 
                                   dz_field='reflectivity')
             radData[tmpRadName] = add_field_to_radar_object(zdr_qc, radData[tmpRadName], field_name='ZDR_qc', units='dB', 
                                   long_name='Differential Reflectivity (Insect/SD-DP Filtered)',
                                   standard_name='Differential Reflectivity (Insect/SD-DP Filtered)', 
                                   dz_field='reflectivity')
             radData[tmpRadName] = add_field_to_radar_object(vel_qc, radData[tmpRadName], field_name='VEL_qc', units='meters_per_second', 
                                   long_name='Radial Velocity (Insect/SD-DP Filtered)',
                                   standard_name='Radial Velocity (Insect/SD-DP Filtered)', 
                                   dz_field='reflectivity')
             radData[tmpRadName] = add_field_to_radar_object(phv_qc, radData[tmpRadName], field_name='PHV_qc', units='ratio', 
                                   long_name='Cross Correlation Ratio (Insect/SD-DP Filtered)',
                                   standard_name='Cross Correlation Ratio (Insect/SD-DP Filtered)', 
                                   dz_field='reflectivity')
             radData[tmpRadName] = add_field_to_radar_object(kdpN, radData[tmpRadName], field_name='KDP', units='deg/km', 
                                   long_name='Specific Differential Phase',
                                   standard_name='Specific Differential Phase', 
                                   dz_field='reflectivity')
             
    
    
    
    # List the fields which will be meshed
    print ('\nThe following fields are included in each file:')
    for radar, value in radData.iteritems():
        print (value.fields.keys())



    # Create the radar composite for each of the included_fields
    # Number of points in the grid (z, y, x)
    # Minimum and maximum grid location (inclusive) in meters for 
    #     the z, y, x coordinates. ( (zMin,zMax),
    #                            (yMin,yMax),
    #                            (xMin,xMax) )
    print ('\nNow meshing radar data {}'.format(time.asctime()))
    mesh_mapped_ALL_3d = pyart.map.grid_from_radars(
        (radData.values()),  
        (4, 601, 601),
        ((1000., 2500.), 
        (-300.*1000., 300.*1000.), 
        (-300.*1000., 300.*1000.)),
        grid_origin=(gridLat, gridLon),
        grid_origin_alt=0.0,
        fields=included_fields,
        copy_field_data=True,
        gridding_algo='map_gates_to_grid')
        
        
    # Make an output directory for the gridded data if 
    # one does not already exist
    # Save the grid to file (named as 'HHMM.nc')
    if not os.path.exists(wrkDir + '/ReflectivityGrids/'):
        os.makedirs(wrkDir + '/ReflectivityGrids/')
    gridPath = wrkDir + '/ReflectivityGrids/' + truGridT + '.nc'
    print ('\nNow writing the output grid. Path: {}'.format(gridPath))
    pyart.io.write_grid(gridPath, mesh_mapped_ALL_3d)

# Define a helper function to easily add new fields to our radar object
def add_field_to_radar_object(field, radar, field_name='field', units='unitless', 
                              long_name='Long Name', standard_name='Standard Name',
                              dz_field='reflectivity'):
    """
    Adds a newly created field to the Py-ART radar object. If reflectivity is a masked array,
    make the new field masked the same as reflectivity.
    """
    fill_value = -32768
    masked_field = np.ma.asanyarray(field)
    masked_field.mask = masked_field == fill_value
    if hasattr(radar.fields[dz_field]['data'], 'mask'):
        setattr(masked_field, 'mask', 
                np.logical_or(masked_field.mask, radar.fields[dz_field]['data'].mask))
        fill_value = radar.fields[dz_field]['_FillValue']
    field_dict = {'data': masked_field,
                  'units': units,
                  'long_name': long_name,
                  'standard_name': standard_name,
                  '_FillValue': fill_value}
    radar.add_field(field_name, field_dict, replace_existing=True)
    return radar    
    
if __name__ == '__main__':
    autoRadCmp(sys.argv[1:])
