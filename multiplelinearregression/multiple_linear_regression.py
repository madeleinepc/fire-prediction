#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 18:45:29 2023

@author: madeleip
"""
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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
import numpy as np

def multiple_linear_regression(df, df_b):

    

    #Remove missing from each row 
    df=df[~np.isnan(df).any(axis=1)]
    
   
    
    #Get predictand 
    Y = df['dNBR']
    df1 = df.drop(columns = {'dNBR','X','Y'})
    #Get predictors 
    #X = df[{'ET_year', 'ET_jan','ESI_year','ESI_jan','Elevation','Aspect','Slope','FWI','Soilm SMAP'}]
    X = df1
    
    #X = df[{'Elevation'}] #Topo only
 
    
    # creating train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.5, random_state=101)
    
    
    
    # creating a regression model
    model = LinearRegression()
    
    
    # fitting the model
    model.fit(X_train,y_train)
    
    # making predictions
    predictions = model.predict(X_test)
    
    # model evaluation
    print(
      'mean_squared_error : ', mean_squared_error(y_test, predictions))
    print(
      'mean_absolute_error : ', mean_absolute_error(y_test, predictions))
    
    
    
    
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_test,predictions)
    line = slope*y_test+intercept
    
    # # #Visualizations 
    plt.plot(y_test, line, 'r', label='y={:.2f}x+{:.2f}'.format(slope,intercept))
    #plt.plot(y_test, y_test, 'k',label='1:1')
    
    plt.scatter(y_test,predictions)#Scatter Density 
    plt.xlabel('observed')
    plt.ylabel('predicted')
    
     
    #plt.annotate("r-squared = {:.3f}".format(r2_score(y_test,predictions)), (np.max(line)*0.5, np.min(line)))
    
    plt.legend(fontsize=14)
    plt.show()
     #End 
     
    print(
       'r^2 value: ', r_value**2)
   
    
    
    # making predictions on Fire B
    
    #Remove missing from each row 
    df_b=df_b[~np.isnan(df_b).any(axis=1)]
    
    try:
        df_b = df_b.drop(columns = 'mask')
    except:
        df_b = df_b
    #Get predictand 
    Y = df_b['dNBR']
    df2 = df_b.drop(columns = {'dNBR','X','Y'})
    #Get predictors 
    X = df2
  
    
    
    
    #X = df_b[{'Elevation'}] #Topo only
    
    predictions = model.predict(X)
    
    
    # model evaluation
    print(
      'mean_squared_error : ', mean_squared_error(Y, predictions))
    print(
      'mean_absolute_error : ', mean_absolute_error(Y, predictions))
    
    
    
    
    
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(Y,predictions)
    line = slope*y_test+intercept
    
    # # #Visualizations 
    plt.plot(y_test, line, 'r', label='y={:.2f}x+{:.2f}'.format(slope,intercept))
    #plt.plot(y_test, y_test, 'k',label='1:1')
    
    plt.scatter(Y,predictions)#Scatter Density 
    plt.xlabel('observed')
    plt.ylabel('predicted')
    
     
    #plt.annotate("r-squared = {:.3f}".format(r2_score(y_test,predictions)), (np.max(line)*0.5, np.min(line)))
    
    plt.legend(fontsize=14)
    plt.show()
    
    print(
      'r^2 value: ', r_value**2)
  
    
    
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    # >>>>>>>>>>>>>>> Plot and Difference Models  for LC <<<<<<<

    
    
    plt.scatter(df_b['X'],df_b['Y'], c = predictions, s =1)
    plt.colorbar()
    plt.title('Predicted')
    plt.show()
    
    plt.scatter(df_b['X'],df_b['Y'], c = Y, s =1)
    plt.colorbar()
    plt.title('Observed')
    plt.show()
    
    plt.scatter(df_b['X'],df_b['Y'], c = Y - predictions, s =1)
    plt.colorbar()
    plt.title('Observed - Predicted')
    plt.show()
    
    #------------------------------------------------------------------------------
    return 