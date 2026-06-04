#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 13:38:15 2022

@author: anna
"""

import numpy as np
from sklearn.decomposition import PCA
from math import exp
import matplotlib.pyplot as plt
import seaborn as sns
from random import random

#uniform clusters centered on each axis
# def cluster(dim, shift = False, shuffle = False, width = 0.5, dist = 1.):
    
#     clust = np.random.uniform(-width/2, width/2, size = (dim*250,dim))

#     for i in range(dim):
#         if shift:
#             r = np.random.uniform(dist-width, dist+width, size = 2)
#         else:
#             r = [dist,dist]
            
#         if shuffle:
#             k = np.random.choice(250)
#         else:
#             k = 125
            
#         for j in range(250):
#             if j < k:
#                 clust[j+(i*250)][i] = np.random.uniform(r[0]-width/2, r[0]+width/2, size = 1)[0]
#             else:
#                 clust[j+(i*250)][i] = np.random.uniform(-r[1]-width/2, -r[1]+width/2, size = 1)[0]
        
#     return np.array(clust)


def cluster(dim, shift = False, shuffle = False, dist = 1.):
    
    '''
    Generate data in any dimension with a cluster on either side of the origin for each dimension
    
    Parameters
    ----------
    dim : int
        number of dimensions
        
    shift : bool, optional
        if False, exactly mirror clusters across origin (negate mu location)
        if True, add noise to mirrored cluster location 
        default is False
        
    shuffle : bool, optional
        if False, 125 points in each cluster
        if True, randomize distribution of 250 points for mirrored clusters
        default is False
                
    dist : float, optional
        upper end of uniform distribution for sampling mu and shift distances
        default is 1

    Returns
    -------
    numpy array
        array of 250*dim vectors of length dim

    '''
    mus = np.random.uniform(-dist,dist,size = (dim, dim))
    var = np.min((1/dim, 0.2))
    data = []
    for i in range(dim):
        mu = mus[i]
        
        if shift:
            r1 = np.random.uniform(-dist, 0, size = dim)
            mu1 = mu +r1
            
            r2 = np.random.uniform(0,dist, size = dim)
            mu2 = (mu*-1)+r2
        else:
            mu1 = mu
            mu2 = mu *-1
            
        if shuffle:
            k = np.random.choice(250)
            clust1 = np.random.normal(mu1,var,size = (k,dim))
            clust2 = np.random.normal(mu2,var,size = (250-k,dim))
        else:
            clust1 = np.random.normal(mu1,var,size = (125,dim))
            clust2 = np.random.normal(mu2,var,size = (125,dim))
        
        data.extend(clust1)
        data.extend(clust2)
        
    return np.array(data)
        
        
# def sphere(dim, radius = 1., cone = False, fill = True, rings = 1):
#     '''

#     Parameters
#     ----------
#     dim : int
#         number of dimensions
        
#     radius : float, optional
#         radius of hypersphere
#         default is 1
        
#     cone_start : float, optional
#         value between 0 and 2, multiplied with pi to define beginning of cone in radians
#         default is 0
        
#     cone_stop : float, optional
#         value between 0 and 2, multiplied with pi to define end of cone in radians--must be greater than cone_start
#         default is 2
        
#     fill : bool, optional
#         if True, randomly sample radius to full in sphere
#         default is True
        
#     rings : int, optional
#         when fill == False, number of equally spaced rings to sample from, with the outermost equal to the radius
#         default is 1

#     Returns
#     -------
#     numpy array
#         array of 250*dim vectors of length dim

#     '''
    
#     if fill:
#             l = np.array([np.random.uniform(0,radius, dim*250)**(1./dim)]).T
#     else:
#         r = np.array([])
#         n = int((dim*250)/rings)
#         for i in range(rings):
#             r = np.concatenate((r,[(i+1)*(radius/rings)]*n))
#             l = np.array([r]).T
    
#     if cone:
#         w = 1/np.sqrt(2)
#         cone_start = -(w/2)
#         cone_stop = +(w/2)
#     else:
#         cone_start = 0
#         cone_stop = 2
    
#     arr1 = np.pi*np.random.uniform(cone_start, cone_stop, [dim*250,1])
#     arr2 = np.pi*np.random.uniform(0, 2, [dim*250,dim-2])
#     arr = np.concatenate((arr1,arr2), axis = 1)
    
#     #arr = np.pi*np.random.uniform(cone_start, cone_stop, [dim*250,dim-1])
    
#     a = np.concatenate((np.array([[2*np.pi]*len(arr)]).T, arr), axis = 1)
#     si = np.sin(a)
#     si.T[0] = 1
#     si = np.cumprod(si, axis = 1)
#     co = np.cos(a)
#     co = np.roll(co, -1)
    
#     sp =  si*co*l
    
#     if cone_stop-cone_start == 2:
#         for i in range(0,len(sp)):
#             np.random.shuffle(sp[i])
    
#     return np.array(sp)

#sample from a normal distribution to populate points on a hypersphere
# def sphere(dim, radius = 1, fill = True, rings = 1):
#     if fill:
#         r = np.array([np.random.uniform(0,radius,size = dim*250)**(1.0/dim)])
#     else:
#         r = np.array([])
#         n = int((dim*250)/rings)
#         for i in range(rings):
#             r = np.concatenate((r,[(i+1)*(radius/rings)]*n))
#         r = np.array([r])
            
#     u = np.random.normal(0,1,(dim*250,dim))
#     norm = np.array([np.linalg.norm(u, axis = 1)])
    
#     return r.T*u/norm.T

#limit normal sphere sampling method to points falling within a radius
# def cone(dim, radius = 1):
#     count = 0
#     r = np.array([np.random.uniform(0,radius,size = dim*250)**(1.0/dim)])
    
#     cone = []
#     while len(cone) < (dim*250):
#         count += 1
#         u = np.random.normal(0,1,dim)
#         norm=np.linalg.norm(u)
#         cos = u/norm.T
#         if np.greater(cos[:-1],-0.5).all() and np.less(cos[:-1],0.5).all()  and cos[-1]< 1 and cos[-1]> 0:
#             cone.append(cos)
#     print(count)        
#     return r.T*cone


def sphere(dim, radius = 1, fill = True, rings = 1, cone = False):
    if fill:
        r = np.array([np.random.uniform(0,radius,size = dim*250)**(1.0/dim)])
    else:
        r = np.array([])
        n = int((dim*250)/rings)
        for i in range(rings):
            r = np.concatenate((r,[(i+1)*(radius/rings)]*n))
        r = np.array([r])
        
    if cone:
        w = 1/np.sqrt(dim)
        r2 = r*np.tan(w)
        r3 = np.array([np.random.uniform(0,r2[0],size = (dim*250))**(1.0/(dim-1))])
        u = np.random.normal(0,1,(dim*250,dim-1))
        norm = np.array([np.linalg.norm(u, axis = 1)])
        sp = r3.T*u/norm.T
        c = np.concatenate((r.T, sp), axis = 1)
        
    else:
        u = np.random.normal(0,1,(dim*250,dim))
        norm = np.array([np.linalg.norm(u, axis = 1)])
        c = r.T*u/norm.T    

    return c