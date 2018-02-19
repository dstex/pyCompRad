import pyart
from matplotlib import pyplot as plt
import numpy as np
import os
import glob

fType = 'png'

files= glob.glob('/data/keeling/a/ammarch2/IOP21/ReflectivityGrids/*.nc')

for i in range(0, len(files)):
	
 
	figFields = ['reflectivity']
	level = 2
	vmin = -8
	vmax = 64

	radarGrid = pyart.io.read_grid(files[i])
	fig = plt.figure(figsize=(13,10))


	display = pyart.graph.GridMapDisplay(radarGrid)
 
	display.plot_basemap(resolution='i')
	
	display.plot_grid(figFields[0],level=level,vmin=vmin,vmax=vmax, cmap=pyart.graph.cm.NWSRef,fig=fig)


	if not os.path.exists('plots/'):
		os.makedirs('plots/')
	fig.savefig('plots/'+display.generate_filename(figFields[0],level=level,ext=fType))    


	plt.close('all')

