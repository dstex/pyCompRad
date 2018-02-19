"""
Guide to modifying the Py-ART gridmapdisplay.py file to include
a few parameters for easy modification of basemap properties.

This file should be located here:
.../anaconda/lib/python3.5/site-packages/pyart/graph/gridmapdisplay.py

"""




class GridMapDisplay(object):
#   |
#   |
#   |
#   |
#   |
    def plot_basemap(
            self, lat_lines=None, lon_lines=None, resolution='l',
            area_thresh=10000, auto_range=True, min_lon=-92, max_lon=-86,
            min_lat=40, max_lat=44,
            gridColor='black',stateBndCol='black',gridDash=[1,1],gridLW=0.5,stateBndLW=0.5,
            ax=None, **kwargs):
        """
        Original looked like this:
            self, lat_lines=None, lon_lines=None, resolution='l',
            area_thresh=10000, auto_range=True, min_lon=-92, max_lon=-86,
            min_lat=40, max_lat=44, ax=None, **kwargs):
        

        Added Parameters
        ----------
        
        gridColor : any color designation
        	Set the color of the lat/lon grid lines
        stateBndCol : any color designation
        	Set the color of the state boundaries
        gridDash : int tuple
        	Used to specify the line dash pattern [points on, points off, 
        	points on, points off, ...]
        gridLW : float
        	Line width of lat/lon grid lines
        stateBndLW : float
        	Line width of state boundaries

        """
        # make basemap
        self._make_basemap(resolution, area_thresh, auto_range,
                           min_lon, max_lon, min_lat, max_lat, ax, **kwargs)

        # parse the parameters
        if lat_lines is None:
            lat_lines = np.arange(30, 46, 1)
        if lon_lines is None:
            lon_lines = np.arange(-110, -75, 1)

        self.basemap.drawcoastlines(linewidth=1.25)
        self.basemap.drawstates(color=stateBndCol,linewidth=stateBndLW)
        self.basemap.drawparallels(
            lat_lines, labels=[True, False, False, False], color=gridColor,
            linewidth=gridLW,dashes=gridDash)
        self.basemap.drawmeridians(
            lon_lines, labels=[False, False, False, True], color=gridColor,
            linewidth=gridLW,dashes=gridDash)
            
            
        """
        Original looked like this:

        if lat_lines is None:
            lat_lines = np.arange(30, 46, 1)
        if lon_lines is None:
            lon_lines = np.arange(-110, -75, 1)

        self.basemap.drawcoastlines(linewidth=1.25)
        self.basemap.drawstates()
        self.basemap.drawparallels(
            lat_lines, labels=[True, False, False, False])
        self.basemap.drawmeridians(
            lon_lines, labels=[False, False, False, True])
        """