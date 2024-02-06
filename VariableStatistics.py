#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 18:29:08 2022

@author: madeleip
"""


#PROJECTS/FIRESENSE/research/codes

import os 
os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/codes')

import scipy
from matplotlib import pyplot as plt
import pandas as pd 
import numpy as np
import pickle 

#%config InlineBackend.figure_format = 'retina'
#Specify RF regression, or RF classification 

#rf_type = 'regression'
rf_type= 'classification'

#Fire Severity Level (For Classification)

#severity = 100 #(low severity)
#severity_black = 1

severity = 270 #(medium severity)
#severity_black = 2
#severity_nm = 1
#severity = 300 #(medium severity)
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


#Johnson
df_johnson = pd.read_pickle("df_johnson.pkl")
LonET_johnson = np.array(pd.read_pickle("df_lon_johnson.pkl"))
LatET_johnson = np.array(pd.read_pickle("df_lat_johnson.pkl"))
mask_johnson = np.array(pd.read_pickle("df_mask_johnson.pkl"))

#Cerro Pelado
df_cerro = pd.read_pickle("df_cerropelado.pkl")
LonET_cerro = np.array(pd.read_pickle("df_lon_cerropelado.pkl"))
LatET_cerro = np.array(pd.read_pickle("df_lat_cerropelado.pkl"))
mask_cerro=np.array(pd.read_pickle("df_mask_cerropelado.pkl"))

# Hermits Peak
df_hermits = pd.read_pickle("df.pkl")
LonET_hermits = np.array(pd.read_pickle("df_lon.pkl"))
LatET_hermits = np.array(pd.read_pickle("df_lat.pkl"))
mask_hermits = np.array(pd.read_pickle("df_mask.pkl"))

# Black
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
# >>>>>>>>>>>>>>> Choose train and prediction fires  <<<<<<<
#    >>>>>>>>>>>>>>>  Training fire data 

#6 Fires 
df = pd.concat((df_doagy, df_johnson, df_cerro, df_black, df_mcbride, df_beartrap, df_hermits, df_cookspeak))

LonET = LonET_hermits
LatET = LatET_hermits 


#    >>>>>>>>>>>>>>>  Prediction fire data 

#This is the fire we are trying to predict

df_b = ((df_hermits))
LonET_b = LonET_hermits
LatET_b = LatET_hermits


#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> PREDICTOR REPRESENTATIVENESS
import seaborn as sns







#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> ALL PREDICTOR TOGETHER


sns.set_style('whitegrid')
l1 = sns.kdeplot(np.array(df_hermits['dNBR'])/1000, bw_method =0.5, label='Hermits', shade = True)
l2 = sns.kdeplot(np.array(df_beartrap['dNBR'])/1000, bw_method =0.5,label='Bear Trap', shade = True)
l3 = sns.kdeplot(np.array(df_cerro['dNBR'])/1000, bw_method =0.5,label='Cerro Pelado', shade = True)
l4 = sns.kdeplot(np.array(df_johnson['dNBR'])/1000, bw_method =0.5,label ='Johnson', shade = True)
l5 = sns.kdeplot(np.array(df_mcbride['dNBR'])/1000, bw_method =0.5,label = 'McBride', shade = True)
l6 = sns.kdeplot(np.array(df_doagy['dNBR'])/1000, bw_method =0.5,label ='Doagy', shade = True)
l6 = sns.kdeplot(np.array(df_black['dNBR'])/1000, bw_method =0.5,label ='Black', shade = True)
l6 = sns.kdeplot(np.array(df_cookspeak['dNBR'])/1000, bw_method =0.5,label ='Cookspeak', shade = True)

plt.title('dNBR')
plt.legend()
plt.show()


fig, axes = plt.subplots(figsize=(12,12), ncols=2, nrows=3)


sns.set_style('whitegrid')
l1 = sns.kdeplot(np.array(df_hermits['ESI_year']), bw_method =0.5, label='Hermits', shade = True, ax=axes[0,0])
l2 = sns.kdeplot(np.array(df_beartrap['ESI_year']), bw_method =0.5,label='Bear Trap', shade = True,ax=axes[0,0])
l3 = sns.kdeplot(np.array(df_cerro['ESI_year']), bw_method =0.5,label='Cerro Pelado', shade = True,ax=axes[0,0])
l4 = sns.kdeplot(np.array(df_johnson['ESI_year']), bw_method =0.5,label ='Johnson', shade = True,ax=axes[0,0])
l5 = sns.kdeplot(np.array(df_mcbride['ESI_year']), bw_method =0.5,label = 'McBride', shade = True,ax=axes[0,0])
l6 = sns.kdeplot(np.array(df_doagy['ESI_year']), bw_method =0.5,label ='Doagy', shade = True,ax=axes[0,0])
l6 = sns.kdeplot(np.array(df_black['ESI_year']), bw_method =0.5,label ='Black', shade = True,ax=axes[0,0])
l6 = sns.kdeplot(np.array(df_cookspeak['ESI_year']), bw_method =0.5,label ='Cookspeak', shade = True,ax=axes[0,0]).set(title='ESI Annual')
#plt.legend()

#plt.show()


sns.set_style('whitegrid')
l1 = sns.kdeplot(np.array(df_hermits['ESI_jan']), bw_method =0.5, label='Hermits', shade = True, ax=axes[0,1])
l2 = sns.kdeplot(np.array(df_beartrap['ESI_jan']), bw_method =0.5,label='Bear Trap', shade = True, ax=axes[0,1])
l3 = sns.kdeplot(np.array(df_cerro['ESI_jan']), bw_method =0.5,label='Cerro Pelado', shade = True, ax=axes[0,1])
l4 = sns.kdeplot(np.array(df_johnson['ESI_jan']), bw_method =0.5,label ='Johnson', shade = True, ax=axes[0,1])
l5 = sns.kdeplot(np.array(df_mcbride['ESI_jan']), bw_method =0.5,label = 'McBride', shade = True, ax=axes[0,1])
l6 = sns.kdeplot(np.array(df_doagy['ESI_jan']), bw_method =0.5,label ='Doagy', shade = True, ax=axes[0,1])
l6 = sns.kdeplot(np.array(df_black['ESI_jan']), bw_method =0.5,label ='Black', shade = True, ax=axes[0,1])
l6 = sns.kdeplot(np.array(df_cookspeak['ESI_jan']), bw_method =0.5,label ='Cookspeak', shade = True, ax=axes[0,1]).set(title = 'b) ESI Before Fire')
plt.legend()
#plt.title('ESI Before Fire')
#plt.show()

sns.set_style('whitegrid')
l1 = sns.kdeplot(np.array(df_hermits['ET_year']), bw_method =0.5, label='Hermits', shade = True, ax=axes[1,0])
l2 = sns.kdeplot(np.array(df_beartrap['ET_year']), bw_method =0.5,label='Bear Trap', shade = True, ax=axes[1,0])
l3 = sns.kdeplot(np.array(df_cerro['ET_year']), bw_method =0.5,label='Cerro Pelado', shade = True, ax=axes[1,0])
l4 = sns.kdeplot(np.array(df_johnson['ET_year']), bw_method =0.5,label ='Johnson', shade = True, ax=axes[1,0])
l5 = sns.kdeplot(np.array(df_mcbride['ET_year']), bw_method =0.5,label = 'McBride', shade = True, ax=axes[1,0])
l6 = sns.kdeplot(np.array(df_doagy['ET_year']), bw_method =0.5,label ='Doagy', shade = True, ax=axes[1,0])
l6 = sns.kdeplot(np.array(df_black['ET_year']), bw_method =0.5,label ='Black', shade = True, ax=axes[1,0])
l6 = sns.kdeplot(np.array(df_cookspeak['ET_year']), bw_method =0.5,label ='Cookspeak', shade = True, ax=axes[1,0]).set(title = 'c) ET Annual')
#plt.legend()
#plt.title('ET Annual')
#plt.show()

sns.set_style('whitegrid')
l1 = sns.kdeplot(np.array(df_hermits['ET_jan']), bw_method =0.5, label='Hermits',shade = True, ax=axes[1,1])
l2 = sns.kdeplot(np.array(df_beartrap['ET_jan']), bw_method =0.5,label='Bear Trap',shade = True, ax=axes[1,1])
l3 = sns.kdeplot(np.array(df_cerro['ET_jan']), bw_method =0.5,label='Cerro Pelado',shade = True, ax=axes[1,1])
l4 = sns.kdeplot(np.array(df_johnson['ET_jan']), bw_method =0.5,label ='Johnson',shade = True, ax=axes[1,1])
l5 = sns.kdeplot(np.array(df_mcbride['ET_jan']), bw_method =0.5,label = 'McBride',shade = True, ax=axes[1,1])
l6 = sns.kdeplot(np.array(df_doagy['ET_jan']), bw_method =0.5,label ='Doagy',shade = True, ax=axes[1,1])
l6 = sns.kdeplot(np.array(df_black['ET_jan']), bw_method =0.5,label ='Black',shade = True, ax=axes[1,1])
l6 = sns.kdeplot(np.array(df_cookspeak['ET_jan']), bw_method =0.5,label ='Cookspeak',shade = True, ax=axes[1,1]).set(title = 'd) ET Before Fire')

#plt.legend()
#plt.title('ET Before Fire')
#plt.show()

# sns.set_style('whitegrid')
# l1 = sns.kdeplot(np.array(df_hermits['SMAP']), bw_method =0.5, label='Hermits',shade = True, ax=axes[2,0])
# l2 = sns.kdeplot(np.array(df_beartrap['SMAP']), bw_method =0.5,label='Bear Trap',shade = True, ax=axes[2,0])
# l3 = sns.kdeplot(np.array(df_cerro['SMAP']), bw_method =0.5,label='Cerro Pelado',shade = True, ax=axes[2,0])
# l4 = sns.kdeplot(np.array(df_johnson['SMAP']), bw_method =0.5,label ='Johnson',shade = True, ax=axes[2,0])
# l5 = sns.kdeplot(np.array(df_mcbride['SMAP']), bw_method =0.5,label = 'McBride',shade = True, ax=axes[2,0])
# l6 = sns.kdeplot(np.array(df_doagy['SMAP']), bw_method =0.5,label ='Doagy',shade = True, ax=axes[2,0])
# l6 = sns.kdeplot(np.array(df_black['SMAP']), bw_method =0.5,label ='Black',shade = True, ax=axes[2,0])
# l6 = sns.kdeplot(np.array(df_cookspeak['SMAP']), bw_method =0.5,label ='Cookspeak',shade = True, ax=axes[2,0])

# plt.legend()
#plt.title('Soil Moisture')
#plt.show()

sns.set_style('whitegrid')
l1 = sns.kdeplot(np.array(df_hermits['VPD']), bw_method =0.5, label='Hermits',shade = True, ax=axes[2,0])
l2 = sns.kdeplot(np.array(df_beartrap['VPD']), bw_method =0.5,label='Bear Trap',shade = True, ax=axes[2,0])
l3 = sns.kdeplot(np.array(df_cerro['VPD']), bw_method =0.5,label='Cerro Pelado',shade = True, ax=axes[2,0])
l4 = sns.kdeplot(np.array(df_johnson['VPD']), bw_method =0.5,label ='Johnson',shade = True, ax=axes[2,0])
l5 = sns.kdeplot(np.array(df_mcbride['VPD']), bw_method =0.5,label = 'McBride',shade = True, ax=axes[2,0])
l6 = sns.kdeplot(np.array(df_doagy['VPD']), bw_method =0.5,label ='Doagy',shade = True, ax=axes[2,0])
l6 = sns.kdeplot(np.array(df_black['VPD']), bw_method =0.5,label ='Black',shade = True, ax=axes[2,0])
l6 = sns.kdeplot(np.array(df_cookspeak['VPD']), bw_method =0.5,label ='Cookspeak',shade = True, ax=axes[2,0]).set(title = 'e) VPD')
#plt.legend()
#plt.title('VPD')
#plt.show()

sns.set_style('whitegrid')
l1 = sns.kdeplot(np.array(df_hermits['TMAX']), bw_method =0.5, label='Hermits',shade = True, ax=axes[2,1])
l2 = sns.kdeplot(np.array(df_beartrap['TMAX']), bw_method =0.5,label='Bear Trap',shade = True, ax=axes[2,1])
l3 = sns.kdeplot(np.array(df_cerro['TMAX']), bw_method =0.5,label='Cerro Pelado',shade = True, ax=axes[2,1])
l4 = sns.kdeplot(np.array(df_johnson['TMAX']), bw_method =0.5,label ='Johnson',shade = True, ax=axes[2,1])
l5 = sns.kdeplot(np.array(df_mcbride['TMAX']), bw_method =0.5,label = 'McBride',shade = True, ax=axes[2,1])
l6 = sns.kdeplot(np.array(df_doagy['TMAX']), bw_method =0.5,label ='Doagy',shade = True, ax=axes[2,1])
l6 = sns.kdeplot(np.array(df_black['TMAX']), bw_method =0.5,label ='Black',shade = True, ax=axes[2,1])
l6 = sns.kdeplot(np.array(df_cookspeak['TMAX']), bw_method =0.5,label ='Cookspeak',shade = True, ax=axes[2,1]).set(title = 'f) TMAX')
#l2 = sns.kdeplot(np.array(df_b['TMAX']), bw =0.5)
#plt.legend(loc='best')
#plt.title('TMAX')
#plt.show()


fig, axes = plt.subplots(figsize=(12,12), ncols=2, nrows=2)


sns.set_style('whitegrid')
l1 = sns.kdeplot(np.array(df_hermits['Elevation']), bw_method =0.5, label='Hermits',shade = True, ax=axes[0,0])
l2 = sns.kdeplot(np.array(df_beartrap['Elevation']), bw_method =0.5,label='Bear Trap',shade = True, ax=axes[0,0])
l3 = sns.kdeplot(np.array(df_cerro['Elevation']), bw_method =0.5,label='Cerro Pelado',shade = True, ax=axes[0,0])
l4 = sns.kdeplot(np.array(df_johnson['Elevation']), bw_method =0.5,label ='Johnson',shade = True, ax=axes[0,0])
l5 = sns.kdeplot(np.array(df_mcbride['Elevation']), bw_method =0.5,label = 'McBride',shade = True, ax=axes[0,0])
l6 = sns.kdeplot(np.array(df_doagy['Elevation']), bw_method =0.5,label ='Doagy',shade = True, ax=axes[0,0])
l6 = sns.kdeplot(np.array(df_black['Elevation']), bw_method =0.5,label ='Black',shade = True, ax=axes[0,0])
l6 = sns.kdeplot(np.array(df_cookspeak['Elevation']), bw_method =0.5,label ='Cookspeak',shade = True, ax=axes[0,0]).set(title = 'a) Elevation')

#plt.legend()
#plt.title('Elevation')
#plt.show()

sns.set_style('whitegrid')
l1 = sns.kdeplot(np.array(df_hermits['Slope']), bw_method =0.5, label='Hermits',shade = True, ax=axes[0,1])
l2 = sns.kdeplot(np.array(df_beartrap['Slope']), bw_method =0.5,label='Bear Trap',shade = True, ax=axes[0,1])
l3 = sns.kdeplot(np.array(df_cerro['Slope']), bw_method =0.5,label='Cerro Pelado',shade = True, ax=axes[0,1])
l4 = sns.kdeplot(np.array(df_johnson['Slope']), bw_method =0.5,label ='Johnson',shade = True, ax=axes[0,1])
l5 = sns.kdeplot(np.array(df_mcbride['Slope']), bw_method =0.5,label = 'McBride',shade = True, ax=axes[0,1])
l6 = sns.kdeplot(np.array(df_doagy['Slope']), bw_method =0.5,label ='Doagy',shade = True, ax=axes[0,1])
l6 = sns.kdeplot(np.array(df_black['Slope']), bw_method =0.5,label ='Black',shade = True, ax=axes[0,1])
l6 = sns.kdeplot(np.array(df_cookspeak['Slope']), bw_method =0.5,label ='Cookspeak',shade = True, ax=axes[0,1]).set(title = 'b) Slope')
#plt.legend()
#plt.title('Slope')
#plt.show()


sns.set_style('whitegrid')
l1 = sns.kdeplot(np.array(df_hermits['Aspect']), bw_method =0.5, label='Hermits',shade = True, ax=axes[1,0])
l2 = sns.kdeplot(np.array(df_beartrap['Aspect']), bw_method =0.5,label='Bear Trap',shade = True, ax=axes[1,0])
l3 = sns.kdeplot(np.array(df_cerro['Aspect']), bw_method =0.5,label='Cerro Pelado',shade = True, ax=axes[1,0])
l4 = sns.kdeplot(np.array(df_johnson['Aspect']), bw_method =0.5,label ='Johnson',shade = True, ax=axes[1,0])
l5 = sns.kdeplot(np.array(df_mcbride['Aspect']), bw_method =0.5,label = 'McBride',shade = True, ax=axes[1,0])
l6 = sns.kdeplot(np.array(df_doagy['Aspect']), bw_method =0.5,label ='Doagy',shade = True, ax=axes[1,0])
l6 = sns.kdeplot(np.array(df_black['Aspect']), bw_method =0.5,label ='Black',shade = True, ax=axes[1,0])
l6 = sns.kdeplot(np.array(df_cookspeak['Aspect']), bw_method =0.5,label ='Cookspeak',shade = True, ax=axes[1,0]).set(title = 'c) Aspect')
plt.legend()
#plt.title('Aspect')
#plt.show()

sns.set_style('whitegrid')
l1 = sns.kdeplot(np.array(df_hermits['FWI']), bw_method =0.5, label='Hermits',shade = True)
l2 = sns.kdeplot(np.array(df_beartrap['FWI']), bw_method =0.5,label='Bear Trap',shade = True)
l3 = sns.kdeplot(np.array(df_cerro['FWI']), bw_method =0.5,label='Cerro Pelado',shade = True)
l4 = sns.kdeplot(np.array(df_johnson['FWI']), bw_method =0.5,label ='Johnson',shade = True)
l5 = sns.kdeplot(np.array(df_mcbride['FWI']), bw_method =0.5,label = 'McBride',shade = True)
l6 = sns.kdeplot(np.array(df_doagy['FWI']), bw_method =0.5,label ='Doagy',shade = True)
l6 = sns.kdeplot(np.array(df_black['FWI']), bw_method =0.5,label ='Black',shade = True)
l6 = sns.kdeplot(np.array(df_cookspeak['FWI']), bw_method =0.5,label ='Cookspeak',shade = True)
plt.legend()
plt.title('FWI')
plt.show()

