#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 19:01:15 2023

@author: madeleip
"""



#PROJECTS/FIRESENSE/research/codes

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

#rf_type= 'classification2'
#fnameD = 'classification2'

#Fire Severity Level (For Classification)

severity = 100 #(medium severity)

#severity = 660 #(high severity)


fname = severity 
os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/workspace/1_week_before')

fnameB = '1_week'

#os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/workspace/1_day_before')
#fnameB = '1_day'


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



os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/codes')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#>>>>>>>>>>>>>>> Choose train and prediction fires  <<<<<<<

#>>>>>>>>>>>>>>>  Training fire data 


df = pd.concat((df_doagy, df_cerro, df_black,  df_mcbride, df_beartrap, df_cookspeak,  df_johnson)) #ALL EXCEPT HERMITS


df = pd.concat((df_doagy, df_johnson,  df_cerro,  df_black)) # 4 Fires

#df = pd.concat(( df_beartrap,  df_johnson, df_black)) #ALL EXCEPT HERMITS


#df = pd.concat((df_black, df_hermits,  df_cerro,  df_cookspeak)) # 4 Fires


# #Rename 
df.rename(columns = {'ET_jan':'ET_nearest', 'ESI_jan':'ESI_nearest'}, inplace = True)

fnameA1 = 'Doagy, Johnson, Cerro, Black'


#To Grid for All Fires
x = np.linspace(df['X'].min(), df['X'].max(), 1000) # create an array of 1000 evenly spaced values between the minimum and maximum of df['X']
y = np.linspace(df['Y'].min(), df['Y'].max(), 1000) # create an array of 1000 evenly spaced values between the minimum and maximum of df['Y']
LonET, LatET = np.meshgrid(x, y) # create a meshgrid from x and y arrays

#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> Eigenvectors for PCNM Prediction  <<<<<<<


##...... COMMENT HERE IF NOT RUNNING SPATIAL AUTOCORRELATION ANALYES 

# #Get x and y coords from df
# xx = df['X']
# yy = df['Y']
# #Subsample every 50th point
# xx = xx[::50]
# yy = yy[::50]
# df =df[::50]

# #Get eigenvectors using pcnm
# eig = spatial_autocorrelation(xx, yy)

# #Use first 10 eigenvectors
# #eig = eig.iloc[:,0:10]

# #include as predictors 
# df=df.assign(PCNM1 = np.array(eig.iloc[:,0]), PCNM2 = np.array(eig.iloc[:,1]), PCNM3 = np.array(eig.iloc[:,2]), PCNM4 = np.array(eig.iloc[:,3]), PCNM5 = np.array(eig.iloc[:,4]), PCNM6 = np.array(eig.iloc[:,5]), PCNM7 = np.array(eig.iloc[:,6]), PCNM8 = np.array(eig.iloc[:,7]), PCNM9 = np.array(eig.iloc[:,8]), PCNM10 = np.array(eig.iloc[:,9]))
# #Reorder cols so x, y last
# df_cols = ['dNBR', 'ET_year', 'ET_jan', 'ESI_year', 'ESI_jan', 'SMAP', 'Elevation', 'Slope', 'Aspect', 'FWI', 'VPD', 'TMAX', 'Land Cover', 'PCNM1', 'PCNM2', 'PCNM3', 'PCNM4', 'PCNM5', 'PCNM6', 'PCNM7', 'PCNM8', 'PCNM9', 'PCNM10', 'X', 'Y']



# #Use first 3 eigenvectors
# df=df.assign(PCNM1 = np.array(eig.iloc[:,0]), PCNM2 = np.array(eig.iloc[:,1]), PCNM3 = np.array(eig.iloc[:,2]))
# df_cols = ['dNBR', 'ET_year', 'ET_nearest', 'ESI_year', 'ESI_nearest', 'SMAP', 'Elevation', 'Slope', 'Aspect', 'FWI', 'VPD', 'TMAX', 'Land Cover', 'PCNM1', 'PCNM2','PCNM3', 'X', 'Y']



# #include as predictors 
# #df=df.assign(PCNM1 = np.array(eig.iloc[:,0]), PCNM2 = np.array(eig.iloc[:,1]))
# #Reorder cols so x, y last
# #df_cols = ['dNBR', 'ET_year', 'ET_jan', 'ESI_year', 'ESI_jan', 'SMAP', 'Elevation', 'Slope', 'Aspect', 'FWI', 'VPD', 'TMAX', 'Land Cover', 'PCNM1', 'PCNM2', 'X', 'Y']



#df = df[df_cols]


##...... COMMENT HERE IF NOT RUNNING SPATIAL AUTOCORRELATION ANALYES 

#    >>>>>>>>>>>>>>>  Prediction fire data 
#This is the fire we are trying to predict

# df_b = ((df_hermits))
# LonET_b = LonET_hermits
# LatET_b = LatET_hermits
# fnameA ='Hermits'

#df_b = (df_black)
#LonET_b = LonET_black
#LatET_b = LatET_black
#fnameA ='Black'

#df_b = (df_cerro)
#LonET_b = LonET_cerro
#LatET_b = LatET_cerro
#fnameA ='Cerro'


#df_b = (df_cookspeak)
#LonET_b = LonET_cookspeak
#LatET_b = LatET_cookspeak
#fnameA ='Cooks'


#df_b = (df_johnson)
#LonET_b = LonET_johnson
#LatET_b = LatET_johnson
#fnameA ='Johnson'


df_b = pd.concat((df_mcbride, df_beartrap, df_cookspeak, df_hermits))
fnameA ='Mcbride, CooksPeak, Beartrap, Hermits'



#df_b= pd.concat((df_doagy, df_mcbride, df_johnson, df_beartrap))

#To Grid for All Fires

x = np.linspace(df_b['X'].min(), df_b['X'].max(), 1000) # create an array of 1000 evenly spaced values between the minimum and maximum of df['X']
y = np.linspace(df_b['Y'].min(), df_b['Y'].max(), 1000) # create an array of 1000 evenly spaced values between the minimum and maximum of df['Y']
LonET_b, LatET_b = np.meshgrid(x, y) # create a meshgrid from x and y arrays


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


#df = df.drop(columns ={'SMAP'})
#df_b=df_b.drop(columns={'SMAP'})
#fnameC = 'no-SMAP'



df = df.drop(columns ={'SMAP', 'FWI','Land Cover'})
df_b=df_b.drop(columns={'SMAP', 'FWI','Land Cover'})
fnameC = 'no-SMAP-FWI'


#df = df.drop(columns ={'VPD','TMAX','SMAP', 'FWI','Land Cover'})
#df_b=df_b.drop(columns={'VPD','TMAX','SMAP', 'FWI','Land Cover'})
#fnameC = 'no-SMAP-FWI-VPD-TMAX'

#df = df.drop(columns ={'VPD','TMAX','ESI_year','ESI_jan','ET_year','ET_jan','SMAP', 'FWI','Land Cover'})
#df_b=df_b.drop(columns={'VPD','TMAX','ESI_year','ESI_jan','ET_year','ET_jan','SMAP','FWI', 'Land Cover'})
#fnameC = 'topo-ONLY'

#df = df.drop(columns ={'ESI_year','ESI_nearest','ET_year','ET_nearest','SMAP', 'FWI','Land Cover'})
#df_b = df_b.drop(columns={'ESI_year','ESI_nearest','ET_year','ET_nearest','SMAP', 'FWI','Land Cover'})
#fnameC = 'No-ECOSTRESS-SMAP-FWI'

#df = df.drop(columns ={'ESI_year','ESI_nearest','ET_year','ET_nearest', 'Land Cover'})
#df_b = df_b.drop(columns={'ESI_year','ESI_nearest','ET_year','ET_nearest','Land Cover'})
#fnameC = 'No-ECOSTRESS'

#THE BEST COMBO:::: vv
#
#
#df = df.drop(columns ={'TMAX','VPD','FWI','SMAP','Elevation','Slope','Aspect','Land Cover'})
#df_b=df_b.drop(columns={'TMAX','VPD','FWI','SMAP','Elevation','Slope','Aspect','Land Cover'})
#fnameC ='no-topo-noweather'


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
  
    R2, mse,Weights,labels,X_train, X_test, y_train, y_test,y_pred,ypredlon,ypredlat,clf  = RandomForestClassification(df,N_train, 1-N_train,name, rf_type)

elif rf_type=='classification2':

    print('running random forest classification2')
    # #For Classification - 5 Classes
    df.loc[df['dNBR'] <= 100,'dNBR'] = 0#
    df.loc[df['dNBR'].between(100,270),'dNBR'] = 1#
    df.loc[df['dNBR'].between(270,440),'dNBR'] = 2#
    df.loc[df['dNBR'].between(440,660),'dNBR'] = 3#
    df.loc[df['dNBR'] > 660,'dNBR'] = 4#
    
    #For Classification - 3 Classes
    #df.loc[df['dNBR'] <= 100, 'dNBR'] = 0# Unburned
    #df.loc[df['dNBR'].between(100,270), 'dNBR'] = 1# Low
    #df.loc[df['dNBR'] > 270, 'dNBR'] = 2# Mod - High
  
    #For Classification - 2 classes
   # df.loc[df['dNBR'] <= 270, 'dNBR'] = 0# Unburned
    #df_b.loc[df_b['dNBR'].between(100,270), 'dNBR'] = 1# Low
    #df.loc[df['dNBR'] > 270, 'dNBR'] = 1# Mod - High


    R2, mse,Weights,labels,X_train, X_test, y_train, y_test,y_pred,ypredlon,ypredlat,clf  = RandomForestClassification(df,N_train, 1-N_train,name, rf_type)



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> Now Tune the MOdel <<<<<<<
#if rf_type=='regression':
 #   test_score, mse,y_pred,rf_tune=TuneRF(X_train,y_train,X_test,y_test)




#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> SCATTER <<<<<<<


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


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> REGRID   <<<<<<<




from regrid.points_to_grid import points_to_grid #Module to regrid points

B = 300 #bin size

obs, pred = points_to_grid(LonET, LatET, ypredlat, ypredlon, y_pred, y_test, B, rf_type)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> WRITE TO GEOTIFF  <<<<<<<

# from writegeotiff import writegeotiff
# os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/output')


# writegeotiff(pred,LonET,LatET,'Prediction' + 'All Fires' + fnameA1+ fnameB + fnameC+ fnameD +'_' + str(fname) + '.tif')
# writegeotiff(obs,LonET,LatET,'Obs' + 'All Fires' + fnameA1 + fnameB + fnameC + fnameD +'_' + str(fname) +'.tif')

# os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/codes')




#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> USE RF model Fire B here    <<<<<<<


#==========================================================================
#Run random forest regression, return R2 and var importance weights 
#Features is an XArray dataset where clumns are variables as 1. dnbr, 2. elev, 3. ET, [...] 4. X, 5.Y
N_train = 0.5

if rf_type == 'regression':
    print('running random forest regression')
    R2,mse, Weights,labels,X_train, X_test, y_train, y_test,y_pred,ypredlon,ypredlat,rf_b  = ApplyRandomForestRegression(df_b,N_train, 1-N_train,name,rf)

elif rf_type == 'classification':
    print('running random forest classification')
    #For Classification
    df_b.loc[df_b['dNBR'] <= severity,'dNBR'] = 0#no burn
    df_b.loc[df_b['dNBR'] > severity,'dNBR'] = 1#burn
   
    R2,mse, Weights,labels,X_train, X_test, y_train, y_test,y_pred,ypredlon,ypredlat,clf_b  = ApplyRandomForestClassification(df_b,N_train, 1-N_train,name,clf, rf_type)

elif rf_type=='classification2':

    print('running random forest classification 4 classes')
    #For Classification - 5 classes
    df_b.loc[df_b['dNBR'] <= 100, 'dNBR'] = 0#
    df_b.loc[df_b['dNBR'].between(100,270), 'dNBR'] = 1#
    df_b.loc[df_b['dNBR'].between(270,440), 'dNBR'] = 2#
    df_b.loc[df_b['dNBR'].between(440,660), 'dNBR'] = 3#
    df_b.loc[df_b['dNBR'] > 660, 'dNBR'] = 4#

    
    #For Classification - 3 classes
   # df_b.loc[df_b['dNBR'] <= 100, 'dNBR'] = 0# Unburned
   # df_b.loc[df_b['dNBR'].between(100,270), 'dNBR'] = 1# Low
   # df_b.loc[df_b['dNBR'] > 270, 'dNBR'] = 2# Mod - High


    #For Classification - 2 classes
    #df_b.loc[df_b['dNBR'] <= 270, 'dNBR'] = 0# Unburned
    #df_b.loc[df_b['dNBR'].between(100,270), 'dNBR'] = 1# Low
    #df_b.loc[df_b['dNBR'] > 270, 'dNBR'] = 1# Mod - High


    R2, mse,Weights,labels,X_train, X_test, y_train, y_test,y_pred,ypredlon,ypredlat,clf_b  = ApplyRandomForestClassification(df_b,N_train, 1-N_train,name,clf, rf_type)

import scipy
 

#------------------------------------------------------------------------------
#       REGRID INPUTS 
#       GRID DATA  Method 2....
#------------------------------------------------------------------------------




from regrid.points_to_grid import points_to_grid #Module to regrid points

B = 300 #bin size

obs, pred = points_to_grid(LonET_b, LatET_b, ypredlat, ypredlon, y_pred, y_test, B, rf_type)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> Plot and Difference Models  for LC <<<<<<<


plt.scatter(ypredlon,ypredlat, c = y_pred, s =1)
plt.colorbar()
plt.title('Predicted' + ' ' + lc_name)
plt.show()

plt.scatter(ypredlon,ypredlat, c = y_test, s =1)
plt.colorbar()
plt.title('Observed' + ' ' + lc_name)
plt.show()

plt.scatter(ypredlon,ypredlat, c = y_test-y_pred, s =1)
plt.colorbar()
plt.title('Observed - Predicted' + ' ' + lc_name)
plt.show()

acc_class = np.where(pred == obs, 0, np.nan)
#------------------------------------------------------------------------------
#                   WRITE PRED/OBS TO GEOTIFF 
#------------------------------------------------------------------------------
from writegeotiff import writegeotiff, writegeotiffNearest
os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/output')


#writegeotiffNearest(pred,LonET_b,LatET_b,'Prediction' + fnameA + fnameB + fnameC+ fnameD +'_' + str(fname) + '.tif')
#writegeotiffNearest(obs,LonET_b,LatET_b,'Obs' + fnameA + fnameB + fnameC + fnameD + '_' + str(fname) +'.tif')
#writegeotiffNearest(acc_class,LonET_b,LatET_b,'Obs_min_pred' + fnameA + fnameB + fnameC + fnameD + '_' + str(fname) +'.tif')


os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/codes')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test, y_pred))

matrix = confusion_matrix(y_test, y_pred)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},cmap=plt.cm.Greens, linewidths=0.2)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()

print(classification_report(y_test, y_pred))









#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

st ='yy'

if st == 'y':
  
    
  
      obs = np.reshape(obs, np.size(obs))
      pred = np.reshape(pred, np.size(pred))
      
      
     # obs = y_test
     # pred = y_pred
      
      obs = np.nan_to_num(obs)
      pred = np.nan_to_num(pred)
      
      id = np.where(obs == 1) #Points in OBS classified as burn
      id_noburn = np.where(pred == 0) #Points in OBS classified as no burn
     
      id_pred = np.where(obs == 1) #Points in prediction classified as burn 
      id_pred_noburn = np.where(pred == 0)#Points in prediction classified as no burn 
      
      
      
      # Percent of predicted burn points accurately classified
      correct = np.squeeze(np.array(np.where(obs[id]==pred[id])))
      per_acc = (len(correct) / len(pred[id])) * 100
      
      
      print(f'Percent Accurately Classified as Burn: {per_acc:>5.3}')
      
      # Percent of all points (burn/no burn) accurately classified
      
      correct = np.squeeze(np.array(np.where(obs == pred)))
      per_acc = (len(correct) / len(obs)) * 100
      
      
      print(f'Percent Accurately Classified: {per_acc:>5.3}')
      
      
      
      
      # Percent of burn omission error 
      incorrect = np.squeeze(np.array(np.where(obs[id] != pred[id])))
      per_acc = (len(incorrect) / len(obs[id])) * 100
      
      
      print(f'Percent Omission: {per_acc:>5.3}')
      
      # Percent of False Positive (comission) 
      incorrect = np.squeeze(np.array(np.where(obs[id_pred] != pred[id_pred])))
      per_acc = (len(incorrect) / len(obs[id_pred])) * 100
      
      
      print(f'False Positive / Error of Commision: {per_acc:>5.3}')
      
