#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:17:11 2023

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
import rioxarray
import geopandas
import xarray
import subprocess

def getFWI(path_to_fwi,path_to_shape, yyyy, mm):
    
    files =os.listdir(path_to_fwi)
    
    #os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/fwi')
    #fn = "wget https://portal.nccs.nasa.gov/datashare/GlobalFWI/v2.0/fwiCalcs.GEOS-5/Default/GPM.LATE.v5/2022/FWI.GPM.LATE.v5.Monthly.Default." + yy + m  + ".nc"
    #runcmd(fn, verbose = True)
    
    
    
    k = 0
    for f in files:
        if f.endswith('.nc') and yyyy in f and mm in f:
              
              print(f)
              fdnbr= path_to_fwi +'/' + f
              
              xds = xarray.open_dataset(fdnbr)
            
              xds = xds['GPM.LATE.v5_FWI']
                  
              xds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
              xds.rio.write_crs("EPSG:4326", inplace=True)
                
              geodf = geopandas.read_file(path_to_shape) 
            
              clipped = xds.rio.clip(geodf.geometry, geodf.crs)
              
              if k == 0:
                  print('create fwi var')
                  fwi = np.zeros(clipped.shape) * np.nan
              fwi = np.vstack((fwi,np.asarray(clipped)))
              k=k+1
    
    array = np.transpose(np.nanmean(fwi, 0))
    longs = clipped['lon']
    lats = clipped['lat']
    
    [lats, longs] = np.meshgrid(np.array(lats), np.array(longs))
    
    return array,lats,longs 


def runcmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

    
