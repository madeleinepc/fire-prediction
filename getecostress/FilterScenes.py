#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:11:51 2022

This script finds number of pixels with data, and filters outscenes with less 
than X-% data
 
@author: madeleip
"""

import numpy as np
import xarray as xr 
def FilterScenes(variable,min_percent):
    
    #Variable - Xarray of ECOSTRESS data (lonxlatxtime) 
    #Filters the data depending on  num cells with data 
    
    #min_percent = percent (0 - 1) of data required 
    
    #Output - xarray of scens with x-% data 
    
    
    
    #Count all elements in scence
    #This is the total number of points 
    
    A_grid=np.asarray(variable[:,:,0])
    eco = np.zeros(A_grid.shape)*np.nan#empty array of zeros to store variables wih reference grid
#       print('First in list')
    yr = 1  
    
    
    grid_y = np.asarray(variable['lat'])
    grid_x = np.asarray(variable['lon'])
    
    
    for j in range(1,len(variable['t'])):
        
        A = variable[:,:,j]
        A=np.asarray(A)
        
        B = int(np.squeeze(variable[:,:,j]['date']))
        numcells = np.count_nonzero(A)
   
        
        #Count all non nan values 
        numcells_data = np.count_nonzero(~np.isnan(A))
        
        
        dataratio = numcells_data/numcells
        
        if dataratio > min_percent:
            
           eco=np.dstack((eco,A))
           yr = np.dstack((yr,B))
    
           
    
    print('storing eco vars')    
    

    
    var = xr.DataArray(eco, dims=['lat','lon','t'],
                             coords ={'lat':grid_y,
                                      'lon':grid_x})   
    
    yeardoy =np.squeeze(yr)
    yeardoy = np.array(yeardoy)
    var["date"] = (['t'],yeardoy)#add the date as a new dim
    
    
    return var, yeardoy 
    
            
    
    
    
    
    
    
    

