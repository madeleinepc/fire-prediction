#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 17:05:57 2023

@author: madeleip
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 16:36:39 2022

@author: madeleip
"""

#WRITE LAT, LON, VALUE NP ARRAY TO GEOTIFF

 
from osgeo import gdal, osr, ogr 
import numpy as np 
import scipy

def getGeoTransform(extent, nlines, ncols):
    resx = (extent[2] - extent[0]) / ncols
    resy = (extent[3] - extent[1]) / nlines
    return [extent[0], resx, 0, extent[3] , 0, -resy]
 

def writegeotiff(data,lon,lat,fname):
    
    #Inputs 
    #Lon -np array of lon
    #Lat - 2d np array of lat
    # data - 2d np array of values 
    #fname - name of HyTES file 
    
    
    #---------------------------------------------------------------------------
    #Regrid to Even Grid??
    
    x1=np.min(lon)
    y1=np.min(lat)
    x2=np.max(lon)
    y2=np.max(lat)
    
    #create even grid from the min/max points 
    
    lnbnd = np.linspace(x1,x2,len(lon[:,0]))
    ltbnd = np.linspace(y1,y2,len(lat[0,:]))
    
    lon2,lat2 = np.meshgrid(lnbnd,ltbnd)
    lon2=np.transpose(lon2);lat2=np.transpose(lat2)
    #Interpolate points to the even grid 
    #interpolating from hytes grid lon,lat to new even grid lon2,lat2
    dataint = scipy.interpolate.griddata((lon.ravel(),lat.ravel()),data.ravel(),(lon2,lat2),'linear');
    
    
    #now use this
    data=np.transpose(np.fliplr(dataint))
    
    lon=lon2
    lat=lat2
    #---------------------------------------------------------------------------
   
    
    
    
    
    
    
    
    
    
    
    
    # Define the data extent (min. lon, min. lat, max. lon, max. lat)
    
    extent = [np.min(lon), np.min(lat), np.max(lon), np.max(lat)] #
    
    # Get GDAL driver GeoTiff
    driver = gdal.GetDriverByName('GTiff')
    
    # Get dimensions
    nlines = data.shape[0]
    ncols = data.shape[1]
    nbands = len(data.shape)
    #data_type = gdal.GDT_Int16
    data_type=gdal.GDT_Float32
    
    
    # Create a temp grid
    #options = ['COMPRESS=JPEG', 'JPEG_QUALITY=80', 'TILED=YES']
    grid_data = driver.Create('grid_data', ncols, nlines, 1, data_type)#, options)
     
    # Write data for each bands
    grid_data.GetRasterBand(1).WriteArray(data)
     
    # Lat/Lon WSG84 Spatial Reference System
    srs = osr.SpatialReference()
    srs.ImportFromProj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
     
    # Setup projection and geo-transform
    grid_data.SetProjection(srs.ExportToWkt())
    grid_data.SetGeoTransform(getGeoTransform(extent, nlines, ncols))
     
    file_name = fname
    print(f'Generated GeoTIFF: {file_name}')
    driver.CreateCopy(file_name, grid_data, 0)  
    
    driver = None
    grid_data = None
     
    # Delete the temp grid
    import os                
    os.remove('grid_data')
    
    
    
def writegeotiffNearest(data,lon,lat,fname):
    
    #Inputs 
    #Lon -np array of lon
    #Lat - 2d np array of lat
    # data - 2d np array of values 
    #fname - name of HyTES file 
    
    
    #---------------------------------------------------------------------------
    #Regrid to Even Grid??
    
    x1=np.min(lon)
    y1=np.min(lat)
    x2=np.max(lon)
    y2=np.max(lat)
    
    #create even grid from the min/max points 
    
    lnbnd = np.linspace(x1,x2,len(lon[:,0]))
    ltbnd = np.linspace(y1,y2,len(lat[0,:]))
    
    lon2,lat2 = np.meshgrid(lnbnd,ltbnd)
    lon2=np.transpose(lon2);lat2=np.transpose(lat2)
    #Interpolate points to the even grid 
    #interpolating from hytes grid lon,lat to new even grid lon2,lat2
    dataint = scipy.interpolate.griddata((lon.ravel(),lat.ravel()),data.ravel(),(lon2,lat2),'nearest');
    
    
    #now use this
    data=np.transpose(np.fliplr(dataint))
    
    lon=lon2
    lat=lat2
    #---------------------------------------------------------------------------
   
    
    
    
    
    
    
    
    
    
    
    
    # Define the data extent (min. lon, min. lat, max. lon, max. lat)
    
    extent = [np.min(lon), np.min(lat), np.max(lon), np.max(lat)] #
    
    # Get GDAL driver GeoTiff
    driver = gdal.GetDriverByName('GTiff')
    
    # Get dimensions
    nlines = data.shape[0]
    ncols = data.shape[1]
    nbands = len(data.shape)
    #data_type = gdal.GDT_Int16
    data_type=gdal.GDT_Float32
    
    
    # Create a temp grid
    #options = ['COMPRESS=JPEG', 'JPEG_QUALITY=80', 'TILED=YES']
    grid_data = driver.Create('grid_data', ncols, nlines, 1, data_type)#, options)
     
    # Write data for each bands
    grid_data.GetRasterBand(1).WriteArray(data)
     
    # Lat/Lon WSG84 Spatial Reference System
    srs = osr.SpatialReference()
    srs.ImportFromProj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
     
    # Setup projection and geo-transform
    grid_data.SetProjection(srs.ExportToWkt())
    grid_data.SetGeoTransform(getGeoTransform(extent, nlines, ncols))
     
    file_name = fname
    print(f'Generated GeoTIFF: {file_name}')
    driver.CreateCopy(file_name, grid_data, 0)  
    
    driver = None
    grid_data = None
     
    # Delete the temp grid
    import os                
    os.remove('grid_data')
    
