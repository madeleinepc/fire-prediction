#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 18:24:54 2022

@author: madeleip
"""


import os
from getancillary.getTif import getTif
from getancillary.getTif import getTifFile
import numpy as np 

def getSMAP(path_to_smap,shape):


 print('getting SMAP...')
 walk_dir = path_to_smap
 # print('walk_dir = ' + walk_dir)
 
 
  
 
 for root, subdirs, files in os.walk(walk_dir):
   #  print('--\nroot = ' + root)
     list_file_path = os.path.join(root, 'my-directory-list.txt')
    # print('list_file_path = ' + list_file_path)
 
 index=0
 
 
 for file in files:#recursive open all files in folder 
     
     if file.endswith('.tif'):
         
             
         soilm,lat_smap,lon_smap=getTifFile(walk_dir+'/'+file,shape)
         #print(soilm)
         if index == 0:
             
             sm = np.zeros(soilm.shape)*np.nan#empty array of zeros to store variables wih reference grid
      #
      
      
         sm = np.dstack((sm,soilm))
         index=index+1



 return sm,lat_smap,lon_smap