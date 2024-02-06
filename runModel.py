#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 08:22:02 2022

@author: madeleip


This program reads in data and runs wildfire severity prediction model 

"""

from getancillary.GetSRTM import GetSRTM
from getecostress.FilterScenes import FilterScenes
from matplotlib import pyplot as plt
import fiona 
from getancillary.getShape import getShape
from getecostress.DataReader import read_ecostress
from getecostress.DataReader import read_ecostress_disalexi
from getecostress.FilterScenes import FilterScenes
import numpy as np 
from getancillary.getDNBR import getDNBR
from regrid.regridinputs import regridinputs
from randomforest.RandomForestRegression import RandomForestRegression 
import elevation
import richdem as rd
import pandas as pd

%config InlineBackend.figure_format = 'retina'



def getVars():

    #------------------------------------------------------------------------------
    #1. Load Path to shapefile of Perimeter fire data 
    path_to_shape = '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/hermitspeak/shapefile'
    #------------------------------------------------------------------------------
    #LOAD SHAPEFILE
    shape = getShape(path_to_shape)
    
    
    
    
    
    
    
    
    
    
    #------------------------------------------------------------------------------
    #2. Path to SRTM topography data 
    path_to_srtm = '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/hermitspeak/srtm'
    
    #------------------------------------------------------------------------------
    #LOAD SRTM DATA CROPPED TO PERIMETER 
    topo,lat_topo,lon_topo=GetSRTM(path_to_srtm,shape)
    #plt.imshow(topo)
    
    
    
    
    
    
    
    
    
    
    #------------------------------------------------------------------------------
    #3. Path to BURN SEVERITY
    path_to_dnbr = '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/hermitspeak/burnseverity'
    #------------------------------------------------------------------------------
    #LOAD dNBR DATA CROPPED TO PERIMETER 
    dnbr,lat_dnbr,lon_dnbr=getDNBR(path_to_dnbr,shape)
    #plt.imshow(topo)
    #plt.pcolor(longs,lats,dnbr,vmin=0,vmax=1000,cmap='pink')
    
    
    
    
    
    
    
    
    
    
    #------------------------------------------------------------------------------
    #4. Path to soil moisture data
    path_to_sm = '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/hermitspeak/ecostress/soilm'
    #------------------------------------------------------------------------------
    #LOAD soil moisture
    
    
    
    
    
    
    #------------------------------------------------------------------------------
    #Path to land cover data 
    #------------------------------------------------------------------------------
    #LOAD Land Cover 
    
    
    
    
    #------------------------------------------------------------------------------
    #5. Path to ECOSTRESS 
    path_to_cloud= '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/hermitspeak/ecostress/cloudmask'
    path_to_ecostress = '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/hermitspeak/ecostress/et'
    
    #------------------------------------------------------------------------------
    #LOAD Land Cover 
    
    #PT-JPL 
    min_percent = 0.50 #minimum percentof data per scene 
    
    A,var,yeardoy_ptjpl=read_ecostress(path_to_ecostress,path_to_cloud)
    et_ptjpl, yeardoy_ptjpl = FilterScenes(var,min_percent) #FILTER OUT DATA WITH LESS THAN x%
    varptjpl=var
    varptjpl_mean=varptjpl.mean(dim='t')
    
    LonET=var.lon
    LatET=var.lat
    
    
    
    
    
    #disALEXI 
    
    path_to_cloud_alexi= '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/hermitspeak/ecostress/uncertainty_disalexi'
    path_to_ecostress_alexi = '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/hermitspeak/ecostress/et_disalexi'
    
    A,var,yeardoy_disalexi=read_ecostress_disalexi(path_to_ecostress_alexi,path_to_cloud_alexi)
    et_disalexi, yeardoy_disalexi = FilterScenes(var,min_percent)#FILTER OUT DATA WITH LESS THAN x%
    
    
    
    
    
    
    
    # #Find scenes that are same. 
    # inddoyptjpl=0
    # inddoydisalexi=0
    # for s in yeardoy_disalexi:
    #     if s in yeardoy_ptjpl:  
            
    #         ind = np.where(yeardoy_ptjpl==s)
    #         ind=np.array(ind)
    #         inddoyptjpl = np.dstack((inddoyptjpl,ind))
            
    #         ind = np.where(yeardoy_disalexi==s)
    #         ind=np.array(ind)
    #         inddoydisalexi = np.dstack((inddoydisalexi,ind))
            
    # inddoyptjpl=np.squeeze(inddoyptjpl)        
    # inddoyptjpl=(inddoyptjpl[2:len(inddoyptjpl)])
    
    # inddoydisalexi=np.squeeze(inddoydisalexi)        
    # inddoydisalexi=(inddoydisalexi[2:len(inddoydisalexi)])    
    
    
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    # >>>>>>>>>>>>>>> REGRID ALL INPUTS TO RESOLUTION /BOUNDARY OF ECOSTRESS <<<<<<<
    
    buff=0
    topo_ecostress, dnbr_ecostress,LonET,LatET = regridinputs(topo,lon_topo,lat_topo,dnbr,lon_dnbr,lat_dnbr,LonET,LatET,buff)
    
    
    #From Topo get Slope and Aspect 
    
    slope = rd.TerrainAttribute(rd.rdarray(topo_ecostress,no_data=0), attrib='slope_riserun')
    aspect = rd.TerrainAttribute(rd.rdarray(topo_ecostress,no_data=0), attrib='aspect')
    
    
    #------------------------------------------------------------------------------
    ## >>>>>>>>>>>>>>> Create Mask  <<<<<<<<<<<<<<<<<<
    #------------------------------------------------------------------------------
    
    
    mask = np.where(topo_ecostress>0,1,np.nan)
    
    
    
    
    #------------------------------------------------------------------------------
    # >>>>>>>>>>>>>>> ReArrange Grids and Return Xarray <<<<<<<
    #------------------------------------------------------------------------------
    
    
    #convert to Pandas Data Frame
    
    #dNBR
    features=np.reshape(np.asarray(dnbr_ecostress),dnbr_ecostress.size)
    #ET ECOSTRESS
    features=np.dstack((features,np.reshape(np.asarray(varptjpl_mean),varptjpl_mean.size)))
    #Topo Elevation
    features=np.dstack((features,np.reshape(np.asarray(topo_ecostress),topo_ecostress.size)))
    #Slope
    features=np.dstack((features,np.reshape(np.asarray(slope),slope.size)))
    #Aspect
    features=np.dstack((features,np.reshape(np.asarray(aspect),aspect.size)))
    #lon
    features=np.dstack((features,np.reshape(np.asarray(LonET),LonET.size)))
    #lat
    features=np.dstack((features,np.reshape(np.asarray(LatET),LatET.size)))
    
    #lat
    features=np.squeeze(features)
        
        
    df = pd.DataFrame(features, columns=["dNBR", "ET_year","Elevation","Slope","Aspect","X","Y"])
    print(df)


return df 











































#Plot ECOSTRESS Scenes Here 


# #PLOT
# plt.rc('image', cmap='YlGn')
# plt.pcolor(var.lon,var.lat,et_disalexi[:,:,inddoydisalexi].mean(dim='t'),vmin=0,vmax=10)
# plt.colorbar()
# plt.title('ET DISALEXI [mm/day]')
# plt.show()

# #Convert to mm/day
# # Watt /m2 = 0.0864 MJ /m2/day
# # MJ /m2/day  =0.408 mm /day 
# # Watt/m2 = 0.0864*0.408 mm/day = 0.035251199999999996 mm/day

# plt.rc('image', cmap='YlGn')
# plt.pcolor(varptjpl.lon,var.lat,et_ptjpl[:,:,inddoyptjpl].mean(dim='t'))
# plt.colorbar()
# plt.title('ET PT-JPL [W/m2]')
# plt.show()



# #Individually see ET images by day 

# for jj in range(0,len(inddoydisalexi)):
#     plt.rc('image', cmap='YlGn')
#     plt.pcolor(var.lon,var.lat,et_disalexi[:,:,inddoydisalexi[jj]],vmin=0,vmax=10)
#     plt.colorbar()
#     plt.xlabel('ET DISALEXI [mm/day]')
#     plt.title(yeardoy_disalexi[inddoydisalexi[jj]])
#     plt.show()
    
    
    
# for jj in range(0,len(inddoyptjpl)):
#     plt.rc('image', cmap='YlGn')
#     plt.pcolor(var.lon,var.lat,et_ptjpl[:,:,inddoydisalexi[jj]]*0.035251199999999996,vmin=0,vmax=10)
#     plt.colorbar()
#     plt.xlabel('ET PTJPL [mm/day]')
#     plt.title(yeardoy_ptjpl[inddoyptjpl[jj]])
#     plt.show()


# topo=np.where(topo==0,np.nan,topo)
# plt.rc('image', cmap='terrain')
# plt.pcolor(lon_topo,lat_topo,topo,vmin=500,vmax=4000)
# plt.colorbar()
# plt.title('SRTM Topography [m]')
# plt.show()






