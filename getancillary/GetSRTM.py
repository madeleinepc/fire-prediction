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

def GetSRTM(path_to_srtm,shape):
    
    files =os.listdir(path_to_srtm)
    
    for f in files:
        if f.endswith('.tif'):
              fsrtm= path_to_srtm +'/' + f
    
    img = rasterio.open(fsrtm)
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
    lats, longs = transform(p1, p2, eastings, northings)
    
    
    return array,lats,longs 



    
    