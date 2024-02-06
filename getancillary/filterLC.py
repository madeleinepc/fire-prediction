#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 16:05:27 2023

@author: madeleip
"""

def filterLC(df, lc_type):
    " Takes in a dataframe object, and filters rows by land cover type from lc_type"
    #Guide to Values 
    
    # 0 - water
    # 1 - Evergreen Needleleaf
    # 2 - Evergreen Broadleaf Forests 
    # 3 -    Deciduous Needleleaf Forests
    # 4 - Deciduous Broadleaf Forests
    # 5 - Mixed Forests
    # 6 - Closed Shrublands
    # 7 - Open shrublands 
    # 8 - Woody savanna
    # 9 - Savanna
    # 10 - Grasslands
    # 11 - Permanent Wetlands
    # 12 - Croplands
    # 13 - Urban and Built
    # 14 - Cropland / natural vegetation mosaic
    # 15 - non vegetated lands
    # 16 - Unclassified 



    # selecting rows based on condition 
    df_filt = df[df['Land Cover'] == lc_type] 
 
    if lc_type == 0:
        lc_name = 'water'
    elif lc_type == 1:
        lc_name = 'Evergreen Needleleaf Forest'
    elif lc_type == 2:
        lc_name = 'Evergreen Broadleaf Forest'
    elif lc_type == 3:
        lc_name = 'Deciduous Needleleaf Forests'
    elif lc_type == 4:
        lc_name = 'Deciduous Broadleaf Forests'
    elif lc_type == 5:
        lc_name = 'Mixed Forests'
    elif lc_type == 6:
         lc_name = 'Closed Shrublands'
    elif lc_type == 7:
         lc_name = 'Open Shrublands'
    elif lc_type == 8:
         lc_name = 'Woody savanna'
    elif lc_type == 9:
         lc_name = 'Savanna'
    elif lc_type == 10:
         lc_name = 'Grasslands'
    elif lc_type ==11:
        lc_name = 'Permanent Wetlands'
    elif lc_type == 12:
         lc_name ='Croplands'
    elif lc_type == 13:
         lc_name ='Urban and Built'
    elif lc_type == 14:
        lc_name = 'Cropland / natural vegetation mosaic'
    elif lc_type ==15:
         lc_name ='non vegetated lands'
    elif lc_type == 16:
         lc_name = 'Unclassified '
    
    #And drop column 
    df_filt = df_filt.drop(columns = 'Land Cover')

    
    
    return df_filt, lc_name