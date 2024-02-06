#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 18:13:09 2023

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

from scipy import stats
from sklearn.metrics import r2_score
from multiplelinearregression.multiple_linear_regression import multiple_linear_regression
#Specify RF regression, or RF classification 

rf_type = 'regression'
#rf_type= 'classification'


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#>> Load Data 

#>>>>>>>>>>>>>>>
#               Load Fire A
#               This is the fire that TRAINS the model
#>>>>>>>>>>>>>>>
#Read PKL data 

df_johnson = pd.read_pickle("df_johnson.pkl")
LonET_johnson = np.array(pd.read_pickle("df_lon_johnson.pkl"))
LatET_johnson = np.array(pd.read_pickle("df_lat_johnson.pkl"))
mask_johnson = np.array(pd.read_pickle("df_mask_johnson.pkl"))



#>>>>>>>>>>>>>>>
#               Load Fire B
#               This is the fire for running prediction
#>>>>>>>>>>>>>>>

#Load Fire B cerro pelado
df_cerro = pd.read_pickle("df_cerropelado.pkl")
LonET_cerro = np.array(pd.read_pickle("df_lon_cerropelado.pkl"))
LatET_cerro = np.array(pd.read_pickle("df_lat_cerropelado.pkl"))
mask_cerro=np.array(pd.read_pickle("df_mask_cerropelado.pkl"))



#Load Fire B
#Read PKL data FOR hermits peak 
df_hermits = pd.read_pickle("df.pkl")
LonET_hermits = np.array(pd.read_pickle("df_lon.pkl"))
LatET_hermits = np.array(pd.read_pickle("df_lat.pkl"))
mask_hermits = np.array(pd.read_pickle("df_mask.pkl"))
#df_b=df_b.drop(columns='Soilm ECOSTRESS')





#Load Fire B
#Read PKL data FOR hermits peak 
df_black = pd.read_pickle("df_black.pkl")
LonET_black = np.array(pd.read_pickle("df_lon_black.pkl"))
LatET_black = np.array(pd.read_pickle("df_lat_black.pkl"))
mask_black =np.array(pd.read_pickle("df_mask_black.pkl"))
#df_b=df_b.drop(columns='Soilm ECOSTRESS')




#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> Choose train and prediction fires  <<<<<<<

#    >>>>>>>>>>>>>>>  Training Fire Data 


df = pd.concat((df_cerro, df_johnson))




#    >>>>>>>>>>>>>>>  Prediction fire data 
df_b = df_hermits


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> If desired, Filter by Land Cover Type<<<<<<<




from getancillary.filterLC import filterLC

# #Filter by evergreen needleleaf(1)

#lc_type = 9 #Savannah
#lc_type = 8 #woody savannah
lc_type = 10 #grassland
#lc_type = 1 #ENF

# #Return data frame with points just corresponding to lc_type
df,lc_name = filterLC(df, lc_type)
df_b,lc_name = filterLC(df_b, lc_type)



# Filter based on dNBR
#lc_name ='All Landcover'
#df = df[df['dNBR'] > 500] 
#df_b = df_b[df_b['dNBR'] > 500] 

#Filter based on topo 
#df = df[df['Elevation'] > 3000]
#df = df[df['Slope'] > 5] 

#df_b = df_b[df_b['Elevation'] > 3000]
#df_b = df_b[df_b['Slope'] >5] #filter

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> Create Multiple Linear Regression MOdel <<<<<<<



#Drop columns here 
#df_b = df_b[{'dNBR','ET_year', 'ET_jan','ESI_year','ESI_jan','Elevation','Aspect','Slope','FWI','Soilm SMAP','X','Y'}]
#df = df[{'dNBR','ET_year', 'ET_jan','ESI_year','ESI_jan','Elevation','Aspect','Slope', 'FWI','Soilm SMAP','X','Y'}]

#df_b = df_b[{'dNBR','ET_year', 'ET_jan','ESI_year','ESI_jan','Elevation','Aspect','Slope','Soilm SMAP','X','Y'}]
#df = df[{'dNBR','ET_year', 'ET_jan','ESI_year','ESI_jan','Elevation','Aspect','Slope', 'Soilm SMAP','X','Y'}]

# df_b = df_b[{'dNBR','ET_year', 'ET_jan','ESI_year','ESI_jan','Elevation','Aspect','Slope','X','Y'}]
# df = df[{'dNBR','ET_year', 'ET_jan','ESI_year','ESI_jan','Elevation','Aspect','Slope','X','Y'}]

#df_b = df_b[{'dNBR','Elevation','Aspect','Slope','X','Y','mask'}]
#df = df[{'dNBR','Elevation','Aspect','Slope','X','Y'}]

#df_b = df_b[{'dNBR','Elevation','Aspect','Slope','X','Y','mask'}]
#df = df[{'dNBR','Elevation','Aspect','Slope','X','Y'}]



multiple_linear_regression(df, df_b)

# from logisticregression.logisticRegression import logisticRegression 
# df.loc[df['dNBR'] <= 1,'dNBR'] = 0#no burn
# df.loc[df['dNBR'] > 1,'dNBR'] = 1#burn
# df_b.loc[df_b['dNBR'] <= 1,'dNBR'] = 0#no burn
# df_b.loc[df_b['dNBR'] > 1,'dNBR'] = 1#burn

# logisticRegression(df, df_b)
