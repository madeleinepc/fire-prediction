#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 12:24:17 2022

A code for cleaning ECOSTRESS files 
Reads in L3 product name, and finds corresponding cloud mask 
@author: madeleip
"""

def clean_ecostress(fname,foldername_cloudmask):
    
    
    
    """create cloud mask 
    
    PARAMETERS
    ----------
    
    input: 
    foldername_cloudmask :Location of L2 cloud mask  
    fname : file name of L3 product 
    
     
       
    RETURNS
    -------
    1) cloud mask 
    1 : no cloud
    nan : cloud 
    """
    walk_dir_mask = foldername_cloudmask
   
    for filesmask in os.walk(walk_dir_mask):
       #  print('--\nroot = ' + root)
         list_file_path = os.path.join(root, 'my-directory-list.txt')
        # print('list_file_path = ' + list_file_path)
    index=0
     
     
    for file in files:#recursive open all files in folder 
         #print(file)
         
         if fname.contains()
         
     
     

  
    
 