#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 18:29:08 2022

@author: madeleip
"""


#PROJECTS/FIRESENSE/research/codes
%config InlineBackend.figure_format = 'retina'

import os 
os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/codes')

#from getecostress.getVars import getVars
from randomforest.RandomForestRegression import RandomForestRegression, ApplyRandomForestRegression,TuneRF, RandomForestClassification, ApplyRandomForestClassification
import scipy
from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np
import pickle 
#import xarray as xr
from getancillary.filterLC import filterLC
from spatial_autocorrelation import spatial_autocorrelation
#%config InlineBackend.figure_format = 'retina'
#Specify RF regression, or RF classification 

rf_type = 'regression'
fnameD ='regression'

#rf_type= 'classification'
#fnameD = 'classification'

#Fire Severity Level (For Classification)

severity = 270 #(medium severity)
#severity = 300 #(medium severity)
#severity = 660 #(high severity)
%config InlineBackend.figure_format = 'retina'

fname = severity 

os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/workspace/1_week_before')

fnameB = '1_week'

#os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/workspace/1_day_before')

#fnameB = '1_day'



#Run SA Analysisis
sa_run ='No' #does not run SA analysis 
#sa_run ='SA' #runs with SA analysis and vars 

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#>> Load Data 

#>>>>>>>>>>>>>>>
#               Load Fire 
#               
#>>>>>>>>>>>>>>>
#Read PKL data 

#For One WEEK BEFORE
#>>>>>>



df_johnson = pd.read_pickle("df_johnson.pkl")
LonET_johnson = np.array(pd.read_pickle("df_lon_johnson.pkl"))
LatET_johnson = np.array(pd.read_pickle("df_lat_johnson.pkl"))
mask_johnson = np.array(pd.read_pickle("df_mask_johnson.pkl"))



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




#Load Fire B
#Read PKL data FOR hermits peak 
df_black = pd.read_pickle("df_black.pkl")
LonET_black = np.array(pd.read_pickle("df_lon_black.pkl"))
LatET_black = np.array(pd.read_pickle("df_lat_black.pkl"))
mask_black =np.array(pd.read_pickle("df_mask_black.pkl"))


#df_black[df_black['dNBR'] == 15] = np.nan
#df_black[df_black['dNBR'] >= severity_black] = severity + 1


#Read PKL data FOR doagy
df_doagy = pd.read_pickle("df_doagy.pkl")
LonET_doagy = np.array(pd.read_pickle("df_lon_doagy.pkl"))
LatET_doagy = np.array(pd.read_pickle("df_lat_doagy.pkl"))
mask_doagy =np.array(pd.read_pickle("df_mask_doagy.pkl"))


#Read for Beartrap 
df_beartrap = pd.read_pickle("df_beartrap.pkl")
LonET_beartrap = np.array(pd.read_pickle("df_lon_beartrap.pkl"))
LatET_beartrap = np.array(pd.read_pickle("df_lat_beartrap.pkl"))
mask_beartrap =np.array(pd.read_pickle("df_mask_beartrap.pkl"))


#Read for McBride 
df_mcbride = pd.read_pickle("df_mcbride.pkl")
LonET_mcbride = np.array(pd.read_pickle("df_lon_mcbride.pkl"))
LatET_mcbride = np.array(pd.read_pickle("df_lat_mcbride.pkl"))
mask_mcbride =np.array(pd.read_pickle("df_mask_mcbride.pkl"))


#Read for Cooks Peak 
df_cookspeak = pd.read_pickle("df_cookspeak.pkl")
LonET_cookspeak = np.array(pd.read_pickle("df_lon_cookspeak.pkl"))
LatET_cookspeak = np.array(pd.read_pickle("df_lat_cookspeak.pkl"))
mask_cookspeak =np.array(pd.read_pickle("df_mask_cookspeak.pkl"))


#Read PKL data FOR Northern NM
# df_nmnorth = pd.read_pickle("df_nmnorth.pkl")
# LonET_nmnorth = np.array(pd.read_pickle("df_lon_nmnorth.pkl"))
# LatET_nmnorth = np.array(pd.read_pickle("df_lat_nmnorth.pkl"))
# mask_nmnorth =np.array(pd.read_pickle("df_mask_nmnorth.pkl"))

# df_nmnorth.loc[df_nmnorth['dNBR'] >= severity_nm, 'dNBR'] = severity + 1
# df_nmnorth.loc[df_nmnorth['dNBR'] != severity + 1, 'dNBR'] = 0


os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/codes')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> Choose train and prediction fires  <<<<<<<

#    >>>>>>>>>>>>>>>  Training fire data 


#6 Fires 
df = pd.concat((df_doagy, df_johnson, df_cerro, df_black, df_mcbride, df_beartrap, df_cookspeak, df_hermits))


#4 Fires 
#df = pd.concat((df_doagy, df_johnson, df_cerro, df_black))


#Rename 
df.rename(columns = {'ET_jan':'ET_nearest', 'ESI_jan':'ESI_nearest'}, inplace = True)


LonET = LonET_hermits
LatET = LatET_hermits 


#To Grid for All Fires
x = np.linspace(df['X'].min(), df['X'].max(), 1000) # create an array of 1000 evenly spaced values between the minimum and maximum of df['X']
y = np.linspace(df['Y'].min(), df['Y'].max(), 1000) # create an array of 1000 evenly spaced values between the minimum and maximum of df['Y']
LonET, LatET = np.meshgrid(x, y) # create a meshgrid from x and y arrays



#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> Eigenvectors for PCNM Prediction  <<<<<<<
pathname = '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/ancillary_vars/'

if sa_run == 'SA':
    
    if os.path.exists(pathname + name + 'varptjpl_mean.nc') == False: #does not exist
        
        #Get x and y coords from df
        xx = df['X']
        yy = df['Y']
        #Subsample every 50th point
        xx = xx[::50]
        yy = yy[::50]
        df =df[::50]
        
        #Get eigenvectors using pcnm
        eig = spatial_autocorrelation(xx, yy)
        
        #Use first 10 eigenvectors
        #eig = eig.iloc[:,0:10]
        
        #include as predictors 
        #df=df.assign(PCNM1 = np.array(eig.iloc[:,0]), PCNM2 = np.array(eig.iloc[:,1]), PCNM3 = np.array(eig.iloc[:,2]), PCNM4 = np.array(eig.iloc[:,3]), PCNM5 = np.array(eig.iloc[:,4]), PCNM6 = np.array(eig.iloc[:,5]), PCNM7 = np.array(eig.iloc[:,6]), PCNM8 = np.array(eig.iloc[:,7]), PCNM9 = np.array(eig.iloc[:,8]), PCNM10 = np.array(eig.iloc[:,9]))
        #Reorder cols so x, y last
        #df_cols = ['dNBR', 'ET_year', 'ET_jan', 'ESI_year', 'ESI_jan', 'SMAP', 'Elevation', 'Slope', 'Aspect', 'FWI', 'VPD', 'TMAX', 'Land Cover', 'PCNM1', 'PCNM2', 'PCNM3', 'PCNM4', 'PCNM5', 'PCNM6', 'PCNM7', 'PCNM8', 'PCNM9', 'PCNM10', 'X', 'Y']
        
        
        
        #Use first 3 eigenvectors
        df=df.assign(PCNM1 = np.array(eig.iloc[:,0]), PCNM2 = np.array(eig.iloc[:,1]), PCNM3 = np.array(eig.iloc[:,2]))
        df_cols = ['dNBR', 'ET_year', 'ET_nearest', 'ESI_year', 'ESI_nearest', 'SMAP', 'Elevation', 'Slope', 'Aspect', 'FWI', 'VPD', 'TMAX', 'Land Cover', 'PCNM1', 'PCNM2','PCNM3', 'X', 'Y']
        
        
        #Save to PCNM folder
        eig.to_netcdf(path = pathname + 'df_allfires_eig.nc')
        
    else:
    #Open Files, if exist    
        print('EIG exists')
    
        eig = xarray.open_dataset(pathname +  'df_allfires_eig.nc')
        eig = eig['__xarray_dataarray_variable__']
        
        df=df.assign(PCNM1 = np.array(eig.iloc[:,0]), PCNM2 = np.array(eig.iloc[:,1]), PCNM3 = np.array(eig.iloc[:,2]))
        df_cols = ['dNBR', 'ET_year', 'ET_nearest', 'ESI_year', 'ESI_nearest', 'SMAP', 'Elevation', 'Slope', 'Aspect', 'FWI', 'VPD', 'TMAX', 'Land Cover', 'PCNM1', 'PCNM2','PCNM3', 'X', 'Y']
       
    
    #SAVE
    
    
    
    #include as predictors 
    #df=df.assign(PCNM1 = np.array(eig.iloc[:,0]), PCNM2 = np.array(eig.iloc[:,1]))
    #Reorder cols so x, y last
    #df_cols = ['dNBR', 'ET_year', 'ET_jan', 'ESI_year', 'ESI_jan', 'SMAP', 'Elevation', 'Slope', 'Aspect', 'FWI', 'VPD', 'TMAX', 'Land Cover', 'PCNM1', 'PCNM2', 'X', 'Y']
    
    
    
    df = df[df_cols]

#    >>>>>>>>>>>>>>>  Prediction fire data 
#This is the fire we are trying to predict

df_b = ((df_hermits))
LonET_b = LonET_hermits
LatET_b = LatET_hermits
fnameA ='Hermits'

df_b = (df_black)
LonET_b = LonET_black
LatET_b = LatET_black
fnameA ='Black'

df_b = (df_cookspeak)
LonET_b = LonET_cookspeak
LatET_b = LatET_cookspeak
fnameA ='Cooks'


df_b = (df_johnson)
LonET_b = LonET_johnson
LatET_b = LatET_johnson
fnameA ='Johnson'


df_b = pd.concat((df_mcbride, df_beartrap, df_cookspeak, df_hermits))
LonET_b = LonET_johnson
LatET_b = LatET_johnson
fnameA ='McBride, Beartrap, Cooks, Hermits'


#Rename 
df_b.rename(columns = {'ET_jan':'ET_nearest', 'ESI_jan':'ESI_nearest'}, inplace = True)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> For ENTIRE NOTH AREA  <<<<<<<
#Drop vars



#df = df.drop(columns ='Land Cover')
#df_b = df_b.drop(columns='Land Cover')
#fnameC = 'all'

#df = df.drop(columns ={'SMAP', 'Land Cover'})
#df_b=df_b.drop(columns={'SMAP', 'Land Cover'})
#fnameC = 'no-SMAP'

#df = df.drop(columns ={'FWI', 'Land Cover'})
#df_b=df_b.drop(columns={'FWI', 'Land Cover'})
#fnameC = 'no-FWI'

# df = df.drop(columns ={'SMAP', 'FWI','Land Cover'})
# df_b=df_b.drop(columns={'SMAP', 'FWI','Land Cover'})
# fnameC = 'no-SMAP-FWI'


#df = df.drop(columns ={'VPD','TMAX','SMAP', 'FWI','Land Cover'})
#df_b=df_b.drop(columns={'VPD','TMAX','SMAP', 'FWI','Land Cover'})
#fnameC = 'no-SMAP-FWI-VPD-TMAX'

df = df.drop(columns ={'ESI_year','ESI_nearest','ET_year','ET_nearest','SMAP', 'FWI','Land Cover'})
df_b=df_b.drop(columns={'ESI_year','ESI_nearest','ET_year','ET_nearest','SMAP','FWI', 'Land Cover'})
fnameC = 'topo-and-weather-ONLY'

#df = df.drop(columns ={'VPD','TMAX','ESI_year','ESI_nearest','ET_year','ET_nearest','SMAP', 'FWI','Land Cover'})
#df_b=df_b.drop(columns={'VPD','TMAX','ESI_year','ESI_nearest','ET_year','ET_nearest','SMAP','FWI', 'Land Cover'})
#fnameC = 'topo-ONLY'

#df = df.drop(columns ={'TMAX','VPD','Elevation','Slope','Aspect','SMAP', 'FWI','Land Cover'})
#df_b=df_b.drop(columns={'TMAX','VPD','Elevation','Slope','Aspect','SMAP', 'FWI','Land Cover'})
#fnameC ='only-eco'

#df = df.drop(columns ={'ESI_year','ESI_nearest','ET_year','ET_nearest','SMAP', 'FWI','Elevation','Slope','Aspect','Land Cover'})
#df_b=df_b.drop(columns={'ESI_year','ESI_nearest','ET_year','ET_nearest','SMAP','FWI', 'Elevation','Slope','Aspect','Land Cover'})
#fnameC = 'weather-ONLY'


#df = df.drop(columns ={'Elevation','Slope','Aspect','Land Cover'})
#df_b=df_b.drop(columns={'Elevation','Slope','Aspect','Land Cover'})

#df = df.drop(columns ={'Elevation','Slope','Aspect','Land Cover','FWI'})
#df_b=df_b.drop(columns={'Elevation','Slope','Aspect','Land Cover','FWI'})

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> Filter by Land Cover Type<<<<<<<





#df = df.drop(columns ={'Soilm SMAP', 'FWI'})
#df_b=df_b.drop(columns={'Soilm SMAP', 'FWI'})

# #Return data frame with points just corresponding to lc_type
#lc_type = 9 #Savannah
#lc_type = 8 #woody savannah
#lc_type = 10 #grassland
#lc_type = 1 #ENF
#lc_type = 7 #shurb
#lc_type ='Whole Area'

#df,lc_name = filterLC(df, lc_type)
#df_b,lc_name = filterLC(df_b, lc_type)



# Filter based on dNBR
lc_name ='All Landcover'

#Filter based on topo 
#df = df[df['Elevation'] > 2500] 
#df_b = df_b[df_b['Elevation'] > 2500] 



#df = df.drop(columns ={'FWI'})
#df_b=df_b.drop(columns={'FWI'})

#df = df.drop(columns ={'Soilm SMAP', 'FWI'})
#df_b=df_b.drop(columns={'Soilm SMAP', 'FWI'})

#df = df.drop(columns ={'VPD','TMAX','Soilm SMAP', 'FWI'})
#df_b=df_b.drop(columns={'VPD','TMAX','Soilm SMAP', 'FWI'})

#df = df.drop(columns ={'VPD','TMAX','ESI_year','ESI_jan','ET_year','ET_jan','Soilm SMAP', 'FWI'})
#df_b=df_b.drop(columns={'VPD','TMAX','ESI_year','ESI_jan','ET_year','ET_jan','Soilm SMAP','FWI'})
#df.head(0)


#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> Run Fire A through RF , output the model.<<<<<<<



#Returns R2, variable weights, training and testing set, prediction, and y/x points, and model (rf)
N_train = 0.5#training percentage 
name='Hermits Peak Fire '
#Run random forest regression, return R2 and var importance weights 
#Features is an XArray dataset where clumns are variables as 1. dnbr, 2. elev, 3. ET, [...] 4. X, 5.Y

if rf_type == 'regression':
    print('running random forest regression')
    R2, mse,Weights,labels,X_train, X_test, y_train, y_test,y_pred,ypredlon,ypredlat,rf  = RandomForestRegression(df,N_train, 1-N_train,name)

elif rf_type=='classification':
    print('running random forest classification')
    #For Classification
    df.loc[df['dNBR'] <= severity,'dNBR'] = 0#
    df.loc[df['dNBR'] > severity,'dNBR'] = 1#
  
    R2, mse,Weights,labels,X_train, X_test, y_train, y_test,y_pred,ypredlon,ypredlat,clf  = RandomForestClassification(df,N_train, 1-N_train,name)






#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> Now Tune the MOdel <<<<<<<
#if rf_type=='regression':
 #   test_score, mse,y_pred,rf_tune=TuneRF(X_train,y_train,X_test,y_test)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> Regrid points to standard grid  <<<<<<<

import scipy
 #REGRID INPUTS  

pred = scipy.interpolate.griddata((ypredlon.ravel(),ypredlat.ravel()),y_pred.ravel(),(LonET,LatET),'nearest');
obs = scipy.interpolate.griddata((ypredlon.ravel(),ypredlat.ravel()),y_test.ravel(),(LonET,LatET),'nearest');
 #add more vars here 




#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> Plot and Difference Models  for LC <<<<<<<


plt.scatter(ypredlon,ypredlat, c = y_pred, s =1)
plt.colorbar()
plt.title('Predicted')
plt.show()

plt.scatter(ypredlon,ypredlat, c = y_test, s =1)
plt.colorbar()
plt.title('Observed')
plt.show()

plt.scatter(ypredlon,ypredlat, c = y_test-y_pred, s =1)
plt.colorbar()
plt.title('Observed - Predicted')
plt.show()


if rf_type == 'regression':
    zi, yi, xi = np.histogram2d(ypredlat, ypredlon, bins=(2000,2000), weights=y_pred)
    counts, _, _ = np.histogram2d(ypredlat, ypredlon, bins=(2000,2000))
    
    zi = zi / counts
    zi = np.ma.masked_invalid(zi)
    
    
    #zi[zi > 0] = 1
    #zi[zi < 1] = np.nan
    
    XX,YY = np.meshgrid(xi[1:len(xi)],yi[1:len(yi)])
    
    pred = scipy.interpolate.griddata((XX.ravel(),YY.ravel()),zi.ravel(),(LonET,LatET),'nearest');
    
    #pred[pred > 0] = 1
    #pred[pred < 1] = np.nan
    
    #Obs Data Method 2....
    
    zi, yi, xi = np.histogram2d(ypredlat, ypredlon, bins=(2000,2000), weights=y_test)
    counts, _, _ = np.histogram2d(ypredlat, ypredlon, bins=(2000,2000))
    
    zi = zi / counts
    zi = np.ma.masked_invalid(zi)
    
    
    #zi[zi > 0] = 1
    #zi[zi < 1] = np.nan
    
    XX,YY = np.meshgrid(xi[1:len(xi)],yi[1:len(yi)])
    
    obs = scipy.interpolate.griddata((XX.ravel(),YY.ravel()),zi.ravel(),(LonET,LatET),'nearest');
    
    #obs[obs > 0] = 1
    #obs[obs < 1] = np.nan
    obs = obs / 1000
    pred = pred / 1000 #To convert dnbr 


from writegeotiff import writegeotiff
os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/output')



writegeotiff(pred,LonET,LatET,'Prediction' + 'All Fires' + fnameB + fnameC+ fnameD +'_' + str(fname) + '.tif')
writegeotiff(obs,LonET,LatET,'Obs' + 'All Fires' + fnameB + fnameC + fnameD +'_' + str(fname) +'.tif')




#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> USE RF model Fire B here    <<<<<<<
















# #Fire B info 
# name='Hermits Peak Fire'



# #==========================================================================
# #Run random forest regression, return R2 and var importance weights 
# #Features is an XArray dataset where clumns are variables as 1. dnbr, 2. elev, 3. ET, [...] 4. X, 5.Y
# N_train = 0.5

# if rf_type == 'regression':
#     print('running random forest regression')
#     R2,mse, Weights,labels,X_train, X_test, y_train, y_test,y_pred,ypredlon,ypredlat,rf  = ApplyRandomForestRegression(df_b,N_train, 1-N_train,name,rf)

# elif rf_type == 'classification':
#     print('running random forest classification')
#     #For Classification
#     df_b.loc[df_b['dNBR'] <= severity,'dNBR'] = 0#no burn
#     df_b.loc[df_b['dNBR'] > severity,'dNBR'] = 1#burn
   
#     R2,mse, Weights,labels,X_train, X_test, y_train, y_test,y_pred,ypredlon,ypredlat,clf  = ApplyRandomForestClassification(df_b,N_train, 1-N_train,name,clf)

# import scipy
 

# #------------------------------------------------------------------------------
# #       REGRID INPUTS 
# #       GRID DATA  Method 2....
# #------------------------------------------------------------------------------



# L1 =  np.array(np.linspace(np.min(LonET_b),np.min(LonET_b),1000))
# L2 =  np.array(np.linspace(np.min(LonET_b),np.max(LonET_b),1000))
# X,Y = np.array(np.meshgrid(L1,L2))
# LonET_b = np.transpose(Y)

# L1 =  np.array(np.linspace(np.min(LatET_b),np.max(LatET_b),1000))
# L2 =  np.array(np.linspace(np.min(LatET_b),np.max(LatET_b),1000))
# X,Y = np.array(np.meshgrid(L2,L1))
# LatET_b = np.flip(Y)



# if rf_type =='classification':

#     zi, yi, xi = np.histogram2d(ypredlat, ypredlon, bins=(200,200), weights=y_pred)
#     counts, _, _ = np.histogram2d(ypredlat, ypredlon, bins=(200,200))
    
#     zi = zi / counts
#     zi = np.ma.masked_invalid(zi)
    
    
#     zi[zi > 0] = 1
#     zi[zi < 1] = np.nan
    
#     XX,YY = np.meshgrid(xi[1:len(xi)],yi[1:len(yi)])
    
#     pred = scipy.interpolate.griddata((XX.ravel(),YY.ravel()),zi.ravel(),(LonET_b,LatET_b),'nearest');
    
#     pred[pred > 0] = 1
#     pred[pred < 1] = np.nan
    
#     #Obs Data Method 2....
    
#     zi, yi, xi = np.histogram2d(ypredlat, ypredlon, bins=(200,200), weights=y_test)
#     counts, _, _ = np.histogram2d(ypredlat, ypredlon, bins=(200,200))
    
#     zi = zi / counts
#     zi = np.ma.masked_invalid(zi)
    
    
#     zi[zi > 0] = 1
#     zi[zi < 1] = np.nan
    
#     XX,YY = np.meshgrid(xi[1:len(xi)],yi[1:len(yi)])
    
#     obs = scipy.interpolate.griddata((XX.ravel(),YY.ravel()),zi.ravel(),(LonET_b,LatET_b),'nearest');
    
#     obs[obs > 0] = 1
#     obs[obs < 1] = np.nan

# if rf_type == 'regression':
    
    
    
    
#     zi, yi, xi = np.histogram2d(ypredlat, ypredlon, bins=(200,200), weights=y_pred)
#     counts, _, _ = np.histogram2d(ypredlat, ypredlon, bins=(200,200))
    
#     zi = zi / counts
#     zi = np.ma.masked_invalid(zi)
#     #zi = np.where(zi == 0, np.nan,zi)
    
#     #zi[zi > 0] = 1
#     #zi[zi < 1] = np.nan
    
#     XX,YY = np.meshgrid(xi[1:len(xi)],yi[1:len(yi)])
    
#     pred = scipy.interpolate.griddata((XX.ravel(),YY.ravel()),zi.ravel(),(LonET_b,LatET_b),'nearest');
    
#     #pred[pred > 0] = 1
#     #pred[pred < 1] = np.nan
    
#     #Obs Data Method 2....
    
#     zi, yi, xi = np.histogram2d(ypredlat, ypredlon, bins=(200,200), weights=y_test)
#     counts, _, _ = np.histogram2d(ypredlat, ypredlon, bins=(200,200))
    
#     zi = zi / counts
#     zi = np.ma.masked_invalid(zi)
    
#     #zi = np.where(zi == 0, np.nan,zi)
    
#     #zi[zi > 0] = 1
#     #zi[zi < 1] = np.nan
    
#     XX,YY = np.meshgrid(xi[1:len(xi)],yi[1:len(yi)])
    
#     obs = scipy.interpolate.griddata((XX.ravel(),YY.ravel()),zi.ravel(),(LonET_b,LatET_b),'nearest');
    
#     #obs[obs > 0] = 1
#     #obs[obs < 1] = np.nan
#     obs = obs / 1000
#     pred = pred / 1000 #To convert dnbr 




# # Create a 2D grid from the x and y vectors
# # ..... FIX 


# #------------------------------------------------------------------------------
# #------------------------------------------------------------------------------
# # >>>>>>>>>>>>>>> Plot and Difference Models  for LC <<<<<<<


# plt.scatter(ypredlon,ypredlat, c = y_pred, s =1)
# plt.colorbar()
# plt.title('Predicted' + ' ' + lc_name)
# plt.show()

# plt.scatter(ypredlon,ypredlat, c = y_test, s =1)
# plt.colorbar()
# plt.title('Observed' + ' ' + lc_name)
# plt.show()

# plt.scatter(ypredlon,ypredlat, c = y_test-y_pred, s =1)
# plt.colorbar()
# plt.title('Observed - Predicted' + ' ' + lc_name)
# plt.show()


# #------------------------------------------------------------------------------
# #                   WRITE PRED/OBS TO GEOTIFF 
# #------------------------------------------------------------------------------


# from writegeotiff import writegeotiff
# os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/output')


# writegeotiff(pred,LonET_b,LatET_b,'Prediction' + fnameA + fnameB + fnameC+ fnameD +'_' + str(fname) + '.tif')
# writegeotiff(obs,LonET_b,LatET_b,'Obs' + fnameA + fnameB + fnameC + fnameD + '_' + str(fname) +'.tif')


# #------------------------------------------------------------------------------
# #------------------------------------------------------------------------------
# #Test... open wit rasterio and check image geotif

# import rasterio
# from rasterio.plot import show
# fp = 'Prediction' + fnameA + fnameB + fnameC + fnameD + '_' + str(fname) +'.tif'
# img = rasterio.open(fp)
# show(img)

# #Test
