#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:42:18 2023

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
import rasterio
from matplotlib import pyplot

def getLC(path_to_lc,shape):
    #This is MOdis LC type 2 
    files =os.listdir(path_to_lc)
    
    for f in files:
        if f.endswith('.tif'):
              fdnbr= path_to_lc +'/' + f
    
    img = rasterio.open(fdnbr)
    
    
    
    #Plot the Raster 
   # pyplot.imshow(img.read(1),vmin=0,vmax=800,cmap='pink')
    
    
    #out_image, out_transform = rasterio.mask.mask(img, shape, crop=True)
    out_image, out_transform = rasterio.mask.mask(img, shape, crop=True)
    
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
    
    #Get rid of fill value 
    array = np.where(array == 255, np.nan, array)
    
    #Guide to Values 
    
    # 0 - water
    # 1 - Evergreen Needleleaf
    # 2 - Evergreen Broadleaf Forests 
    # 3 -    Deciduous Needleleaf Forests
    # 4 - Deciduous Broadleaf Forests
    # 5 - Mixed Forests
    # 6 - Closed Shrublands
    # 7 - Open shrublands 
    # 8 - Woody savanna
    # 9 - Savanna
    # 10 - Grasslands
    # 11 - Permanent Wetlands
    # 12 - Croplands
    # 13 - Urban and Built
    # 14 - Cropland / natural vegetation mosaic
    # 15 - non vegetated lands
    # 16 - Unclassified 
    
    
    
    return array,lats,longs 



    
    