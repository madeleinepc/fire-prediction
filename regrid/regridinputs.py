#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 15:13:26 2022

@author: madeleip
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 12:03:25 2022

@author: madeleip
"""

import scipy
from datetime import datetime,date 
import numpy as np 

def regridinputs(topo,lon_topo,lat_topo,dnbr,lon_dnbr,lat_dnbr,soilm,lon_soilm,lat_soilm,smap,lon_smap,lat_smap,lon_lc,lat_lc, lc,lon_fwi,lat_fwi,fwi, lon_vpd,lat_vpd,vpd,lon_tmax,lat_tmax,tmax,esiptjpl_mean,esiptjpl_nearest,LonESI,LatESI,LonET,LatET,buff):
    
    #gridded lat lon for ECOST
    (LonET,LatET)=np.meshgrid(LonET,LatET,copy=False)
    
    
    #NDVI
    #NDSI
    #Albedo
    #LonOut
    #LatOut
    #sATM
    #LonHyf
    #LatHyf
    
    
    maxLat = np.max(LatET);
    minLat = np.min(LatET);
    maxLon = np.max(LonET);
    minLon = np.min(LonET);
    
    
    #REGRID INPUTS from SENTINEL 
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    # Crop Topo
    I1 = np.argwhere((lat_topo[:,0]< np.min(np.min(LatET))-buff) | (lat_topo[:,0]> np.max(np.max(LatET))+buff)); # 0.5 degrees offset
    I2 = np.argwhere((lon_topo[0,:]< np.min(np.min(LonET))-buff) | (lon_topo[0,:]> np.max(np.max(LonET))+buff)); # 0.5 degrees offset
    

    I1=list(np.squeeze(I1))
    I2=list(np.squeeze(I2))
    
    
    
    #CROP based on indices
    #lat=np.delete(lat[:,0],I1)
    
    lat_topo =np.delete(lat_topo,I1,0)
    lat_topo =np.delete(lat_topo,I2,1)
    lat_topo=np.squeeze(lat_topo)
    #lat(:,I2) = [];
    #lon=np.delete(lon[0,:],I2) 
    
    
    lon_topo =np.delete(lon_topo,I1,0)
    lon_topo =np.delete(lon_topo,I2,1)
    lon_topo=np.squeeze(lon_topo)
   
    #SUBSET  TOPO 
    topo=np.delete(topo,I1,0)
    topo = np.delete(topo,I2,1)
     
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    # Crop dNBR
    I1 = np.argwhere((lat_dnbr[:,0]< np.min(np.min(LatET))-buff) | (lat_dnbr[:,0]> np.max(np.max(LatET))+buff)); # 0.5 degrees offset
    I2 = np.argwhere((lon_dnbr[0,:]< np.min(np.min(LonET))-buff) | (lon_dnbr[0,:]> np.max(np.max(LonET))+buff)); # 0.5 degrees offset
    

    I1=list(np.squeeze(I1))
    I2=list(np.squeeze(I2))
    
    
    
    #CROP based on indices
    #lat=np.delete(lat[:,0],I1)
    
    lat_dnbr =np.delete(lat_dnbr,I1,0)
    lat_dnbr =np.delete(lat_dnbr,I2,1)
    lat_dnbr=np.squeeze(lat_dnbr)
    
    #lat(:,I2) = [];
    #lon=np.delete(lon[0,:],I2) 
    
    
    lon_dnbr =np.delete(lon_dnbr,I1,0)
    lon_dnbr =np.delete(lon_dnbr,I2,1)
    lon_dnbr=np.squeeze(lon_dnbr)
   
    #SUBSET  TOPO 
    dnbr=np.delete(dnbr,I1,0)
    dnbr = np.delete(dnbr,I2,1)

     
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    # Crop soil
    I1 = np.argwhere((lat_soilm[:,0]< np.min(np.min(LatET))-buff) | (lat_soilm[:,0]> np.max(np.max(LatET))+buff)); # 0.5 degrees offset
    I2 = np.argwhere((lon_soilm[0,:]< np.min(np.min(LonET))-buff) | (lon_soilm[0,:]> np.max(np.max(LonET))+buff)); # 0.5 degrees offset
    

    I1=list(np.squeeze(I1))
    I2=list(np.squeeze(I2))
    
    
    
    #CROP based on indices
    #lat=np.delete(lat[:,0],I1)
    
    lat_soilm =np.delete(lat_soilm,I1,0)
    lat_soilm =np.delete(lat_soilm,I2,1)
    lat_soilm=np.squeeze(lat_soilm)
    
    #lat(:,I2) = [];
    #lon=np.delete(lon[0,:],I2) 
    
    
    lon_soilm =np.delete(lon_soilm,I1,0)
    lon_soilm =np.delete(lon_soilm,I2,1)
    lon_soilm=np.squeeze(lon_soilm)
   
    #SUBSET  sm 
    soilm=np.delete(soilm,I1,0)
    soilm = np.delete(soilm,I2,1)

     
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 


 #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

 #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 
     # Crop SMAP
    I1 = np.argwhere((lat_smap[:,0]< np.min(np.min(LatET))-buff) | (lat_smap[:,0]> np.max(np.max(LatET))+buff)); # 0.5 degrees offset
    I2 = np.argwhere((lon_smap[0,:]< np.min(np.min(LonET))-buff) | (lon_smap[0,:]> np.max(np.max(LonET))+buff)); # 0.5 degrees offset
     
    
    #I1=list(np.squeeze(I1))
    #I2=list(np.squeeze(I2))
     
     
     
     #CROP based on indices
     #lat=np.delete(lat[:,0],I1)
     
    lat_smap =np.delete(lat_smap,I1,0)
    lat_smap =np.delete(lat_smap,I2,1)
    lat_smap=np.squeeze(lat_smap)
     
     #lat(:,I2) = [];
     #lon=np.delete(lon[0,:],I2) 
     
     
    lon_smap =np.delete(lon_smap,I1,0)
    lon_smap =np.delete(lon_smap,I2,1)
    lon_smap=np.squeeze(lon_smap)
    
     #SUBSET  sm 
    smap=np.delete(smap,I1,0)
    smap= np.delete(smap,I2,1)

  #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  
      # Crop LC
    # I1 = np.argwhere((lat_lc[:,0]< np.min(np.min(LatET))-buff) | (lat_lc[:,0]> np.max(np.max(LatET))+buff)); # 0.5 degrees offset
    # I2 = np.argwhere((lon_lc[0,:]< np.min(np.min(LonET))-buff) | (lon_lc[0,:]> np.max(np.max(LonET))+buff)); # 0.5 degrees offset
      
     
    #  #I1=list(np.squeeze(I1))
    #  #I2=list(np.squeeze(I2))
      
      
      
    #   #CROP based on indices
    #   #lat=np.delete(lat[:,0],I1)
      
    # lat_lc =np.delete(lat_lc,I1,0)
    # lat_lc =np.delete(lat_lc,I2,1)
    # lat_lc=np.squeeze(lat_lc)
      
    #   #lat(:,I2) = [];
    #   #lon=np.delete(lon[0,:],I2) 
      
      
    # on_lc =np.delete(lon_lc,I1,0)
    # lon_lc =np.delete(lon_lc,I2,1)
    # lon_lc=np.squeeze(lon_lc)
     
    #   #SUBSET  sm 
    # lc=np.delete(lc,I1,0)
    # lc= np.delete(lc,I2,1)

 #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

 #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

 #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 
    (LonESI,LatESI)=np.meshgrid(LonESI,LatESI,copy=False)
 
     # Crop ESI
    I1 = np.argwhere((LatESI[:,0]< np.min(np.min(LatET))-buff) | (LatESI[:,0]> np.max(np.max(LatET))+buff)); # 0.5 degrees offset
    I2 = np.argwhere((LonESI[0,:]< np.min(np.min(LonET))-buff) | (LonESI[0,:]> np.max(np.max(LonET))+buff)); # 0.5 degrees offset
     
    
    #I1=list(np.squeeze(I1))
    #I2=list(np.squeeze(I2))
     
     
     
     #CROP based on indices
     #lat=np.delete(lat[:,0],I1)
     
    LatESI =np.delete(LatESI,I1,0)
    LatESI =np.delete(LatESI,I2,1)
    LatESI=np.squeeze(LatESI)
     
     #lat(:,I2) = [];
     #lon=np.delete(lon[0,:],I2) 
     
     
    LonESI =np.delete(LonESI,I1,0)
    LonESI =np.delete(LonESI,I2,1)
    LonESI=np.squeeze(LonESI)
    
     #SUBSET  sm 
    esi_year=np.delete(esiptjpl_mean,I1,0)
    esi_year= np.delete(esiptjpl_mean,I2,1)
    
    esi_nearest=np.delete(esiptjpl_nearest,I1,0)
    esi_nearest= np.delete(esiptjpl_nearest,I2,1)

  
 #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>







     
   #Sample over coarser grid 
    dnbr = dnbr[::4,::4]
    topo=topo[::4,::4]
    
    
    
    soilm=soilm[::4,::4]
    #smap=smap[::4,::4]
    esi_year=esi_year[::4,::4]
    esi_nearest=esi_nearest[::4,::4]
    
    # NDSI=NDSI[::4,::4]
    lon_topo=lon_topo[::4,::4]
    lat_topo=lat_topo[::4,::4]
    
    
    lon_dnbr=lon_dnbr[::4,::4]
    lat_dnbr=lat_dnbr[::4,::4]
   
    lon_soilm=lon_soilm[::4,::4]
    lat_soilm=lat_soilm[::4,::4]
    
    #lon_lc=lon_lc[::4,::4]
    #lat_lc=lat_lc[::4,::4]
    
    
    
    LonESI=LonESI[::4,::4]
    LatESI=LatESI[::4,::4]
    #lon_smap=lon_smap[::4,::4]
    #lat_smap=lat_smap[::4,::4]
   
    print('interpolating')
    #REGRID INPUTS  
    topo_ecostress = scipy.interpolate.griddata((lon_topo.ravel(),lat_topo.ravel()),topo.ravel(),(LonET,LatET),'linear');
    dnbr_ecostress = scipy.interpolate.griddata((lon_dnbr.ravel(),lat_dnbr.ravel()),dnbr.ravel(),(LonET,LatET),'linear');
    soilm_ecostress = scipy.interpolate.griddata((lon_soilm.ravel(),lat_soilm.ravel()),soilm.ravel(),(LonET,LatET),'linear');
    smap = scipy.interpolate.griddata((lon_smap.ravel(),lat_smap.ravel()),smap.ravel(),(LonET,LatET),'nearest');
    lc = scipy.interpolate.griddata((lon_lc.ravel(),lat_lc.ravel()),lc.ravel(),(LonET,LatET),'nearest');
    fwi = scipy.interpolate.griddata((lon_fwi.ravel(),lat_fwi.ravel()),fwi.ravel(),(LonET,LatET),'linear');

    vpd = scipy.interpolate.griddata((lon_vpd.ravel(),lat_vpd.ravel()),vpd.ravel(),(LonET,LatET),'linear');
    tmax = scipy.interpolate.griddata((lon_tmax.ravel(),lat_tmax.ravel()),tmax.ravel(),(LonET,LatET),'linear');




    esi_year = scipy.interpolate.griddata((LonESI.ravel(),LatESI.ravel()),esi_year.ravel(),(LonET,LatET),'linear');
    esi_nearest = scipy.interpolate.griddata((LonESI.ravel(),LatESI.ravel()),esi_nearest.ravel(),(LonET,LatET),'linear');
    
    
    
    #add more vars here 
         
       
    
    
       
    
    
    
    return topo_ecostress, dnbr_ecostress,soilm_ecostress,smap, lc, fwi, vpd, tmax, esi_year,esi_nearest,LonET,LatET







def regridinputs2(topo,lon_topo,lat_topo,dnbr,lon_dnbr,lat_dnbr,smap,lon_smap,lat_smap,lon_lc,lat_lc, lc, lon_fwi,lat_fwi,fwi, lon_vpd,lat_vpd, vpd, lon_tmax,lat_tmax, tmax, esiptjpl_mean,esiptjpl_nearest,LonESI,LatESI,LonET,LatET,buff):
    
    #gridded lat lon for ECOST
    (LonET,LatET)=np.meshgrid(LonET,LatET,copy=False)
    
    
    #NDVI
    #NDSI
    #Albedo
    #LonOut
    #LatOut
    #sATM
    #LonHyf
    #LatHyf
    
    
    maxLat = np.max(LatET);
    minLat = np.min(LatET);
    maxLon = np.max(LonET);
    minLon = np.min(LonET);
    
    
    #REGRID INPUTS from SENTINEL 
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    # Crop Topo
    I1 = np.argwhere((lat_topo[:,0]< np.min(np.min(LatET))-buff) | (lat_topo[:,0]> np.max(np.max(LatET))+buff)); # 0.5 degrees offset
    I2 = np.argwhere((lon_topo[0,:]< np.min(np.min(LonET))-buff) | (lon_topo[0,:]> np.max(np.max(LonET))+buff)); # 0.5 degrees offset
    

    I1=list(np.squeeze(I1))
    I2=list(np.squeeze(I2))
    
    
    
    #CROP based on indices
    #lat=np.delete(lat[:,0],I1)
    
    lat_topo =np.delete(lat_topo,I1,0)
    lat_topo =np.delete(lat_topo,I2,1)
    lat_topo=np.squeeze(lat_topo)
    #lat(:,I2) = [];
    #lon=np.delete(lon[0,:],I2) 
    
    
    lon_topo =np.delete(lon_topo,I1,0)
    lon_topo =np.delete(lon_topo,I2,1)
    lon_topo=np.squeeze(lon_topo)
   
    #SUBSET  TOPO 
    topo=np.delete(topo,I1,0)
    topo = np.delete(topo,I2,1)
     
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    # Crop dNBR
    I1 = np.argwhere((lat_dnbr[:,0]< np.min(np.min(LatET))-buff) | (lat_dnbr[:,0]> np.max(np.max(LatET))+buff)); # 0.5 degrees offset
    I2 = np.argwhere((lon_dnbr[0,:]< np.min(np.min(LonET))-buff) | (lon_dnbr[0,:]> np.max(np.max(LonET))+buff)); # 0.5 degrees offset
    

    I1=list(np.squeeze(I1))
    I2=list(np.squeeze(I2))
    
    
    
    #CROP based on indices
    #lat=np.delete(lat[:,0],I1)
    
    lat_dnbr =np.delete(lat_dnbr,I1,0)
    lat_dnbr =np.delete(lat_dnbr,I2,1)
    lat_dnbr=np.squeeze(lat_dnbr)
    
    #lat(:,I2) = [];
    #lon=np.delete(lon[0,:],I2) 
    
    
    lon_dnbr =np.delete(lon_dnbr,I1,0)
    lon_dnbr =np.delete(lon_dnbr,I2,1)
    lon_dnbr=np.squeeze(lon_dnbr)
   
    #SUBSET  TOPO 
    dnbr=np.delete(dnbr,I1,0)
    dnbr = np.delete(dnbr,I2,1)

     
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
  
     
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 


 #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

 #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 
     # Crop SMAP
    # I1 = np.argwhere((lat_smap[:,0]< np.min(np.min(LatET))-buff) | (lat_smap[:,0]> np.max(np.max(LatET))+buff)); # 0.5 degrees offset
    # I2 = np.argwhere((lon_smap[0,:]< np.min(np.min(LonET))-buff) | (lon_smap[0,:]> np.max(np.max(LonET))+buff)); # 0.5 degrees offset
     
    
    # #I1=list(np.squeeze(I1))
    # #I2=list(np.squeeze(I2))
     
     
     
    #  #CROP based on indices
    #  #lat=np.delete(lat[:,0],I1)
     
    # lat_smap =np.delete(lat_smap,I1,0)
    # lat_smap =np.delete(lat_smap,I2,1)
    # lat_smap=np.squeeze(lat_smap)
     
    #  #lat(:,I2) = [];
    #  #lon=np.delete(lon[0,:],I2) 
     
     
    # lon_smap =np.delete(lon_smap,I1,0)
    # lon_smap =np.delete(lon_smap,I2,1)
    # lon_smap=np.squeeze(lon_smap)
    
    #  #SUBSET  sm 
    # smap=np.delete(smap,I1,0)
    # smap= np.delete(smap,I2,1)

  #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  
      # Crop LC
    # buff = 0
    # I1 = np.argwhere((lat_lc[:,0]< np.min(np.min(LatET))-buff) | (lat_lc[:,0]> np.max(np.max(LatET))+buff)); # 0.5 degrees offset
    # I2 = np.argwhere((lon_lc[0,:]< np.min(np.min(LonET))-buff) | (lon_lc[0,:]> np.max(np.max(LonET))+buff)); # 0.5 degrees offset
      
     
    #  #I1=list(np.squeeze(I1))
    #  #I2=list(np.squeeze(I2))
      
      
      
    #   #CROP based on indices
    #   #lat=np.delete(lat[:,0],I1)
      
    # lat_lc =np.delete(lat_lc,I1,0)
    # lat_lc =np.delete(lat_lc,I2,1)
    # lat_lc=np.squeeze(lat_lc)
      
    #   #lat(:,I2) = [];
    #   #lon=np.delete(lon[0,:],I2) 
      
      
    # lon_lc =np.delete(lon_lc,I1,0)
    # lon_lc =np.delete(lon_lc,I2,1)
    # lon_lc=np.squeeze(lon_lc)
     
    #   #SUBSET  sm 
    # lc=np.delete(lc,I1,0)
    # lc= np.delete(lc,I2,1)

 #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

 #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

 #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

 #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

 #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     
    (LonESI,LatESI)=np.meshgrid(LonESI,LatESI,copy=False)
 
     # Crop ESI
    I1 = np.argwhere((LatESI[:,0]< np.min(np.min(LatET))-buff) | (LatESI[:,0]> np.max(np.max(LatET))+buff)); # 0.5 degrees offset
    I2 = np.argwhere((LonESI[0,:]< np.min(np.min(LonET))-buff) | (LonESI[0,:]> np.max(np.max(LonET))+buff)); # 0.5 degrees offset
     
    
    #I1=list(np.squeeze(I1))
    #I2=list(np.squeeze(I2))
     
     
     
     #CROP based on indices
     #lat=np.delete(lat[:,0],I1)
     
    LatESI =np.delete(LatESI,I1,0)
    LatESI =np.delete(LatESI,I2,1)
    LatESI=np.squeeze(LatESI)
     
     #lat(:,I2) = [];
     #lon=np.delete(lon[0,:],I2) 
     
     
    LonESI =np.delete(LonESI,I1,0)
    LonESI =np.delete(LonESI,I2,1)
    LonESI=np.squeeze(LonESI)
    
     #SUBSET  sm 
    esi_year=np.delete(esiptjpl_mean,I1,0)
    esi_year= np.delete(esiptjpl_mean,I2,1)
    
    esi_nearest=np.delete(esiptjpl_nearest,I1,0)
    esi_nearest= np.delete(esiptjpl_nearest,I2,1)

  
 #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    vpd = np.asarray(vpd)
    vpd = np.transpose(vpd)
    tmax = np.asarray(tmax)
    tmax = np.transpose(tmax)
     
   #Sample over coarser grid 
    dnbr = dnbr[::4,::4]
    topo=topo[::4,::4]
    
    
    
     #smap=smap[::4,::4]
    esi_year=esi_year[::4,::4]
    esi_nearest=esi_nearest[::4,::4]
    
    # NDSI=NDSI[::4,::4]
    lon_topo=lon_topo[::4,::4]
    lat_topo=lat_topo[::4,::4]
    
    
    lon_dnbr=lon_dnbr[::4,::4]
    lat_dnbr=lat_dnbr[::4,::4]
   
    
    LonESI=LonESI[::4,::4]
    LatESI=LatESI[::4,::4]
    #lon_smap=lon_smap[::4,::4]
    #lat_smap=lat_smap[::4,::4]
   
    print('interpolating ...')
    #REGRID INPUTS  
    topo_ecostress = scipy.interpolate.griddata((lon_topo.ravel(),lat_topo.ravel()),topo.ravel(),(LonET,LatET),'nearest');
    dnbr_ecostress = scipy.interpolate.griddata((lon_dnbr.ravel(),lat_dnbr.ravel()),dnbr.ravel(),(LonET,LatET),'nearest');
    smap = scipy.interpolate.griddata((lon_smap.ravel(),lat_smap.ravel()),smap.ravel(),(LonET,LatET),'nearest');
    lc = scipy.interpolate.griddata((lon_lc.ravel(),lat_lc.ravel()),lc.ravel(),(LonET,LatET),'nearest');
    
    fwi = scipy.interpolate.griddata((lon_fwi.ravel(),lat_fwi.ravel()),fwi.ravel(),(LonET,LatET),'nearest');
    
    vpd = scipy.interpolate.griddata((lon_vpd.ravel(),lat_vpd.ravel()),vpd.ravel(),(LonET,LatET),'nearest');
    tmax = scipy.interpolate.griddata((lon_tmax.ravel(),lat_tmax.ravel()),tmax.ravel(),(LonET,LatET),'nearest');

    esi_year = scipy.interpolate.griddata((LonESI.ravel(),LatESI.ravel()),esi_year.ravel(),(LonET,LatET),'linear');
    esi_nearest = scipy.interpolate.griddata((LonESI.ravel(),LatESI.ravel()),esi_nearest.ravel(),(LonET,LatET),'linear');
    
    
    
    #add more vars here 
         
       
    
    
       
    
    
    
    return topo_ecostress, dnbr_ecostress,smap, lc, fwi,vpd, tmax, esi_year,esi_nearest,LonET,LatET