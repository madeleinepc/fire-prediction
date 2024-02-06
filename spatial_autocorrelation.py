#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 14:19:27 2023

@author: madeleip
"""

from sklearn.manifold import MDS
from scipy.spatial import distance_matrix
from skbio.stats.ordination import pcoa
import numpy as np
from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination import pcoa
import numpy as np
from scipy.spatial.distance import cdist
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from scipy.spatial.distance import pdist, squareform
from haversine import haversine


def spatial_autocorrelation(x, y):
# Reads in x y coords and returns PCNM vectors for use as predictors in 
# statistical models 
    
    # X Y into coords
     
    coordinates = np.array([x, y])
    
    # Transpose
    #coordinates = np.transpose(coordinates)

    # Calculate the pairwise distances using the Haversine formula for Lon and Lat coords
    distances = pdist([(radians(lat), radians(lon)) for lat, lon in coordinates.T], haversine)

    # Convert the pairwise distances to a square distance matrix
    dist_matrix = squareform(distances)

    
    
   
    #========================== FIX ! ========================== 
   
    
   # Calculate the pairwise distances between the coordinates
    #dis = distance_matrix(coordinates, coordinates)
    
    #Faster???
    #dis = cdist(coordinates,coordinates)

    #Get Eigenvectors
    eigenvectors = pcnm(dist_matrix)

    return eigenvectors

def pcnm(dis, threshold=None, w=None, dist_ret=False):
   """This function computed classical PCNM by the principal coordinate
   analysis of a truncated distance matrix. These are commonly used to
   transform (spatial) distances to rectangular data that suitable for
   constrained ordination or regression.
   
   
   dis : a distance matrix
   
   threshold : a threshold value or truncation distance. 
   (If missing, minimum distance giving connected netwrok will be used. This is 
   found as longest distance in min spacnning tree of dis}
    
   w : weights for rows
   
   dist_ret: return distances used to calculate PCNMs "
   
   
   
   
   """
   
   # Convert distance matrix to skbio DistanceMatrix object
   dis = DistanceMatrix(dis)

   # Calculate principal coordinates analysis
   pcoa_results = pcoa(dis)

   # Extract the eigenvectors
   eigenvectors = pcoa_results.samples

   # Truncate eigenvectors based on threshold
   if threshold is not None:
        eigenvectors = eigenvectors[:, dis.data <= threshold]

   # Scale eigenvectors to unit norm
   eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

   # Return distances used to calculate the PCNMs if requested
   if dist_ret:
        return (eigenvectors, dis.data)
   else:
        return eigenvectors