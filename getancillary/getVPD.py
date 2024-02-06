#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:15:25 2023

@author: madeleip
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
import rioxarray
import geopandas
import xarray
import subprocess
import datetime


def getVPD(path_to_vpd,path_to_shape, yyyy, mm, dd):
    
    files =os.listdir(path_to_vpd)
    
    #Get Julian Date
    jd = datetime.date(int(yyyy),int(mm),int(dd)).toordinal() - datetime.date(int(yyyy),1,1).toordinal()

    for f in files:
        if f.endswith('.nc') and yyyy in f:
              
             
              vpd= path_to_vpd +'/' + f
              
              xds = xarray.open_dataset(vpd)
              xds = xds['mean_vapor_pressure_deficit']
                  
              xds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
              xds.rio.write_crs("EPSG:4326", inplace=True)
                
              geodf = geopandas.read_file(path_to_shape)
            
              clipped = xds.rio.clip(geodf.geometry, geodf.crs)
              
    #GET VPD For DAY FIRE START          
    array = clipped.isel(day = jd)
    longs = clipped['lon']
    lats = clipped['lat']
    
    [lats, longs] = np.meshgrid(np.array(lats), np.array(longs))
    
    return array,lats,longs 



    
