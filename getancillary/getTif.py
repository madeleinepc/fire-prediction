#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 07:41:25 2022

@author: madeleip
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 13:32:58 2022

@author: madeleip
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 14:30:59 2022

@author: madeleip

Load SRTM data


"""
import os
import rasterio 
from rasterio.plot import show
from pyproj import Proj, transform
import numpy as np
from affine import Affine
import fiona
import rasterio.mask
from matplotlib import pyplot

def getTif(path_to,shape):
    
    files =os.listdir(path_to)
    
    for f in files:
        if f.endswith('.tif'):
              fdnbr= path_to +'/' + f
    
    img = rasterio.open(fdnbr)
    
    
    
    #Plot the Raster 
   # pyplot.imshow(img.read(1),vmin=0,vmax=800,cmap='pink')
    
    
    out_image, out_transform = rasterio.mask.mask(img, shape, invert=True)
    out_image=np.squeeze(out_image)
    
    
    #T0 = img.transform  # upper-left pixel corner affine transform
    T0 =out_transform
    p1 = Proj(img.crs)
    #array = img.read(1) # pixel values
    array = out_image 
    
    
    # All rows and columns
    cols, rows = np.meshgrid(np.arange(array.shape[1]), np.arange(array.shape[0]))

    # Get affine transform for pixel centres
    T1 = T0 * Affine.translation(0.5, 0.5)
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: (c, r) * T1

    # All eastings and northings (there is probably a faster way to do this)
    eastings, northings = np.vectorize(rc2en, otypes=[np.float, np.float])(rows, cols)

    # Project all longitudes, latitudes
    p2 = Proj(proj='latlong',datum='WGS84')
    longs, lats = transform(p1, p2, eastings, northings)
    
    
    return array,lats,longs 







def getTifFile(f,shape):
    
 
    
    img = rasterio.open(f)
    
    
    
    #Plot the Raster 
   # pyplot.imshow(img.read(1),vmin=0,vmax=800,cmap='pink')
    
    
    out_image, out_transform = rasterio.mask.mask(img, shape, invert=True)
    out_image=np.squeeze(out_image)
    
    
    #T0 = img.transform  # upper-left pixel corner affine transform
    T0 =out_transform
    p1 = Proj(img.crs)
    #array = img.read(1) # pixel values
    array = out_image 
    
    
    # All rows and columns
    cols, rows = np.meshgrid(np.arange(array.shape[1]), np.arange(array.shape[0]))

    # Get affine transform for pixel centres
    T1 = T0 * Affine.translation(0.5, 0.5)
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: (c, r) * T1

    # All eastings and northings (there is probably a faster way to do this)
    eastings, northings = np.vectorize(rc2en, otypes=[float, float])(rows, cols)

    # Project all longitudes, latitudes
    p2 = Proj(proj='latlong',datum='WGS84')
    longs, lats = transform(p1, p2, eastings, northings)
    
    
    return array,lats,longs 



    
    