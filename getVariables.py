#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:38:54 2023

@author: madeleip
"""



#PROJECTS/FIRESENSE/research/codes

import os 
os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/codes')

from getecostress.getVars import getVars
from randomforest.RandomForestRegression import RandomForestRegression, ApplyRandomForestRegression,TuneRF, RandomForestClassification, ApplyRandomForestClassification
import scipy
from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np
import pickle 
import xarray as xr
from getancillary.filterLC import filterLC

#%config InlineBackend.figure_format = 'retina'


#Specify RF regression, or RF classification 
#------------------------------------------------------------------------------
#                   STEP 1 GET VARIABLES FOR EACH FIRE 
#------------------------------------------------------------------------------




#Specify Fire name 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#>>                 Fire A (build the model)
name='hermitspeak'
yyyy = '2022'
mm = '04'
dd = '06'

#Get WEEK Before 
mm = '03'
dd ='30'

# # #Get the Variables in Data Frame Format for Each Fire 
df,mask,LonET,LatET=getVars(name, yyyy, mm, dd)

os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/workspace/1_week_before')


# # # # #Save Variable 
df.to_pickle("df.pkl")

# # # # # #Convert Lon Lat to df and save 
df_lat = pd.DataFrame(np.array(LatET))
df_lon = pd.DataFrame(np.array(LonET))
df_mask = pd.DataFrame(mask)


# # # # # # Saving the objects:
df_lat.to_pickle("df_lat.pkl")
df_lon.to_pickle("df_lon.pkl")
df_mask.to_pickle("df_mask.pkl")



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#>>                 GET DATA FOR Fire B 

name ='johnson'
yyyy = '2021'
mm = '05'
dd = '20'


#Get WEEK Before 
mm = '05'
dd ='13'

# # #Get the Variables in Data Frame Format for Each Fire 
df,mask,LonET,LatET=getVars(name, yyyy, mm, dd)

os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/workspace/1_week_before')



# # #Save Variables 
df.to_pickle("df_johnson.pkl")


# # # # #Convert Lon Lat to df and save 
df_lat_johnson = pd.DataFrame(np.array(LatET))
df_lon_johnson = pd.DataFrame(np.array(LonET))
df_mask_johnson = pd.DataFrame(mask)


# # # # # Saving the objects:
df_lat_johnson.to_pickle("df_lat_johnson.pkl")
df_lon_johnson.to_pickle("df_lon_johnson.pkl")
df_mask_johnson.to_pickle("df_mask_johnson.pkl")

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#>>                 GET DATA FOR Fire C 

name='cerropelado'
yyyy = '2022'
mm ='04'
dd = '22'

#Get WEEK Before 
mm = '04'
dd ='15'

# # #Get the Variables in Data Frame Format for Each Fire 
df,mask,LonET,LatET=getVars(name, yyyy, mm, dd)

os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/workspace/1_week_before')


# #Save Variable 
df.to_pickle("df_cerropelado.pkl")


# # # #Convert Lon Lat to df and save 
df_lat_cerropelado = pd.DataFrame(np.array(LatET))
df_lon_cerropelado = pd.DataFrame(np.array(LonET))
df_mask_cerropelado = pd.DataFrame(mask)


# # # # Saving the objects:
df_lat_cerropelado.to_pickle("df_lat_cerropelado.pkl")
df_lon_cerropelado.to_pickle("df_lon_cerropelado.pkl")
df_mask_cerropelado.to_pickle("df_mask_cerropelado.pkl")


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#>>                 GET DATA FOR Fire C 




name='black'
yyyy = '2022'
mm ='05'
dd = '13'


#Get WEEK Before 
mm = '05'
dd ='16'

# # #Get the Variables in Data Frame Format for Each Fire 
df,mask,LonET,LatET=getVars(name, yyyy, mm, dd)

os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/workspace/1_week_before')


# # # #Get the Variables in Data Frame Format for Each Fire 
# df,mask,LonET,LatET=getVars(name, yyyy, mm,dd)

#df['ET_year'] = df['ET_year'].where(df['ET_year'] > 1000, np.nan)

# # # #Save Variable 
df.to_pickle("df_black.pkl")


# # # # # #Convert Lon Lat to df and save 
df_lat_black = pd.DataFrame(np.array(LatET))
df_lon_black = pd.DataFrame(np.array(LonET))
df_mask_black = pd.DataFrame(mask)


# # # # Saving the objects:
df_lat_black.to_pickle("df_lat_black.pkl")
df_lon_black.to_pickle("df_lon_black.pkl")
df_mask_black.to_pickle("df_mask_black.pkl")




#Specify Fire name 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#>>                 Fire A (build the model)
name='doagy'
yyyy = '2021'
mm = '05'
dd = '14'
# # #Get the Variables in Data Frame Format for Each Fire 
# df,mask,LonET,LatET=getVars(name, yyyy, mm, dd)

#Get WEEK Before 
mm = '05'
dd ='7'

# # #Get the Variables in Data Frame Format for Each Fire 
df,mask,LonET,LatET=getVars(name, yyyy, mm, dd)


os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/workspace/1_week_before')

# # # #Save Variable 
df.to_pickle("df_doagy.pkl")

# # # # # #Convert Lon Lat to df and save 
df_lat_doagy = pd.DataFrame(np.array(LatET))
df_lon_doagy = pd.DataFrame(np.array(LonET))
df_mask_doagy = pd.DataFrame(mask)


# # # # # Saving the objects:
df_lat_doagy.to_pickle("df_lat_doagy.pkl")
df_lon_doagy.to_pickle("df_lon_doagy.pkl")
df_mask_doagy.to_pickle("df_mask_doagy.pkl")



#Specify Fire name 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#>>                 Fire A (build the model)
name='nm_north'
yyyy = '2022'
mm = '04'
dd = '6'
# # #Get the Variables in Data Frame Format for Each Fire 
# df,mask,LonET,LatET=getVars(name, yyyy, mm, dd)

#Get WEEK Before 
mm = '03'
dd ='30'

# # #Get the Variables in Data Frame Format for Each Fire 
df,mask,LonET,LatET=getVars(name, yyyy, mm, dd)


os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/workspace/1_week_before')

# # # #Save Variable 
df.to_pickle("df_nmnorth.pkl")

# # # # # #Convert Lon Lat to df and save 
df_lat_nmnorth = pd.DataFrame(np.array(LatET))
df_lon_nmnorth = pd.DataFrame(np.array(LonET))
df_mask_nmnorth = pd.DataFrame(mask)


# # # # # Saving the objects:
df_lat_nmnorth.to_pickle("df_lat_nmnorth.pkl")
df_lon_nmnorth.to_pickle("df_lon_nmnorth.pkl")
df_mask_nmnorth.to_pickle("df_mask_nmnorth.pkl")





#Specify Fire name 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#>>                 Bear TRAP FIRE

name='beartrap'
yyyy = '2022'
mm = '05'
dd = '1'
# # #Get the Variables in Data Frame Format for Each Fire 
# df,mask,LonET,LatET=getVars(name, yyyy, mm, dd)

#Get WEEK Before 
mm = '04'
dd ='27'

# # #Get the Variables in Data Frame Format for Each Fire 
df,mask,LonET,LatET=getVars(name, yyyy, mm, dd)


os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/workspace/1_week_before')

# # # #Save Variable 
df.to_pickle("df_beartrap.pkl")

# # # # # #Convert Lon Lat to df and save 
df_lat_beartrap = pd.DataFrame(np.array(LatET))
df_lon_beartrap = pd.DataFrame(np.array(LonET))
df_mask_beartrap = pd.DataFrame(mask)


# # # # # Saving the objects:
df_lat_beartrap.to_pickle("df_lat_beartrap.pkl")
df_lon_beartrap.to_pickle("df_lon_beartrap.pkl")
df_mask_beartrap.to_pickle("df_mask_beartrap.pkl")




#Specify Fire name 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#>>                 MCBRIDE

name='mcbride'
yyyy = '2022'
mm = '04'
dd = '12'
# # #Get the Variables in Data Frame Format for Each Fire 
# df,mask,LonET,LatET=getVars(name, yyyy, mm, dd)

#Get WEEK Before 
mm = '04'
dd ='5'

# # #Get the Variables in Data Frame Format for Each Fire 
df,mask,LonET,LatET=getVars(name, yyyy, mm, dd)


os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/workspace/1_week_before')

# # # #Save Variable 
df.to_pickle("df_mcbride.pkl")

# # # # # #Convert Lon Lat to df and save 
df_lat_mcbride = pd.DataFrame(np.array(LatET))
df_lon_mcbride = pd.DataFrame(np.array(LonET))
df_mask_mcbride = pd.DataFrame(mask)


# # # # # Saving the objects:
df_lat_mcbride.to_pickle("df_lat_mcbride.pkl")
df_lon_mcbride.to_pickle("df_lon_mcbride.pkl")
df_mask_mcbride.to_pickle("df_mask_mcbride.pkl")






#Specify Fire name 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#>>                COOKS PEAK

name='cookspeak'
yyyy = '2022'
mm = '04'
dd = '17'
# # #Get the Variables in Data Frame Format for Each Fire 
# df,mask,LonET,LatET=getVars(name, yyyy, mm, dd)

#Get WEEK Before 
mm = '04'
dd ='10'

# # #Get the Variables in Data Frame Format for Each Fire 
df,mask,LonET,LatET=getVars(name, yyyy, mm, dd)


os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/workspace/1_week_before')

# # # #Save Variable 

#Multiply df['dNBR'] by 1000....
df['dNBR']=df['dNBR']*1000


df.to_pickle("df_cookspeak.pkl")

# # # # # #Convert Lon Lat to df and save 
df_lat_cookspeak = pd.DataFrame(np.array(LatET))
df_lon_cookspeak = pd.DataFrame(np.array(LonET))
df_mask_cookspeak = pd.DataFrame(mask)


# # # # # Saving the objects:
df_lat_cookspeak.to_pickle("df_lat_cookspeak.pkl")
df_lon_cookspeak.to_pickle("df_lon_cookspeak.pkl")
df_mask_cookspeak.to_pickle("df_mask_cookspeak.pkl")










#Specify Fire name 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#>>                 Fire A (build the model)
#name = 'nm_north'
#yyyy = '2021'
#mm = '04'
#dd = '01'
# # #Get the Variables in Data Frame Format for Each Fire
#df, mask, LonET, LatET = getVars(name, yyyy, mm, dd)



# os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/workspace')

# # # # #Save Variable 
# df.to_pickle("df_poso.pkl")

# # # # # # #Convert Lon Lat to df and save 
# df_lat_poso = pd.DataFrame(np.array(LatET))
# df_lon_poso = pd.DataFrame(np.array(LonET))
# df_mask_poso = pd.DataFrame(mask)


# # # # # # Saving the objects:
# df_lat_poso.to_pickle("df_lat_poso.pkl")
# df_lon_poso.to_pickle("df_lon_poso.pkl")
# df_mask_poso.to_pickle("df_mask_poso.pkl")
