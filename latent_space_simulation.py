#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 09:48:54 2022

@author: anna
"""
import numpy as np
#%%
#create 10K 100dim vectors with uniform dist between [-1,1]

uni = []
for i in range(10000):
    v = np.random.uniform(-1,1,100)
    uni.append(v)

#%%
#create 10K 100dim vectors with uniform dist between [-1,1] for first 75 dims and uniform dist between [0,1] for last 25 dims

uni_75 = []
for i in range(10000):
    v1 = np.random.uniform(-1,1,75)
    v2 = np.random.uniform(0,1,25)
    v = np.concatenate((v1,v2))
    uni_75.append(v)
    
#%%
#create 10K 100dim vectors with uniform dist between [-1,1] for first 50 dims and uniform dist between [0,1] for last 50 dims

uni_50 = []
for i in range(10000):
    v1 = np.random.uniform(-1,1,50)
    v2 = np.random.uniform(0,1,50)
    v = np.concatenate((v1,v2))
    uni_50.append(v)
    
#%%
#create 10K 100dim vectors with uniform dist between [-1,1] for first 25 dims and uniform dist between [0,1] for last 75 dims

uni_25 = []
for i in range(10000):
    v1 = np.random.uniform(-1,1,25)
    v2 = np.random.uniform(0,1,75)
    v = np.concatenate((v1,v2))
    uni_25.append(v)
    
#%%
#create 10K 100dim vectors with uniform dist between [-1,1] for first 50 dims and uniform dist between [-1,0] for last 50 dims

uni_neg = []
for i in range(10000):
    v1 = np.random.uniform(-1,1,50)
    v2 = np.random.uniform(-1,0,50)
    v = np.concatenate((v1,v2))
    uni_neg.append(v)
    
#%%
#create 10K 100dim vectors with 10dims at uniform dist between [-1,1], 10dims at uniform dist between [-0.1,0.1] repeating 

uni_spread_cluster = []
for i in range(10000):
    v = np.array([])
    for j in range(5):
        v1 = np.random.uniform(-1,1,10)
        v2 = np.random.uniform(-.1,.1,10)
        v = np.concatenate((v,v1,v2))
    uni_spread_cluster.append(v)

#%%
#create 10K 100dim vectors with 10dims at uniform dist between [-1,1], 10dims at 0 repeating 

uni_zero_cluster = []
for i in range(10000):
    v = np.array([])
    for j in range(5):
        v1 = np.random.uniform(-10,10,10)
        v2 = np.array([0]*10)
        v = np.concatenate((v,v1,v2))
    uni_zero_cluster.append(v)
#%%
#create 10K 100dim vectors with 10dims at uniform dist between [-1,1], 10dims at uniform dist between [0,1] repeating 

uni_shift_cluster = []
for i in range(10000):
    v = np.array([])
    for j in range(5):
        v1 = np.random.uniform(-1,1,10)
        v2 = np.random.uniform(0,1,10)
        v = np.concatenate((v,v1,v2))
    uni_shift_cluster.append(v)

#%%
#create 10K 100dim vectors with uniform dist between [-1,1] for first 50 dims and uniform dist between [0,1] for last 50 dims

uni_50_zero = []
for i in range(10000):
    v1 = np.random.uniform(-1,1,50)
    v2 = np.array([0]*50)
    v = np.concatenate((v1,v2))
    uni_50_zero.append(v)

#%%
#create 10K 100dim vectors with uniform dist between [0,1] for first 50 dims and uniform dist between [-1,0] for last 50 dims

uni_opp = []
for i in range(10000):
    v1 = np.random.uniform(0,1,50)
    v2 = np.random.uniform(-1,0,50)
    v = np.concatenate((v1,v2))
    uni_opp.append(v)
    
#%%
#create 10K 100dim vectors ~N(0,1)

norm = []
for i in range(10000):
    v = np.random.normal(size = 100)
    norm.append(v)
    
#%%
#create 10K 100dim vectors ~N(0,1) first 50 dims and ~N(0,5) for last 50 dims

norm_sd_spread = []
for i in range(10000):
    v1 = np.random.normal(0,1,50)
    v2 = np.random.normal(0,5,50)
    v = np.concatenate((v1,v2))
    norm_sd_spread.append(v)
    
#%%
#create 10K 100dim vectors ~N(0,1) first 50 dims and ~N(-1,1) for last 50 dims

norm_mu_shift = []
for i in range(10000):
    v1 = np.random.normal(0,1,50)
    v2 = np.random.normal(-1,1,50)
    v = np.concatenate((v1,v2))
    norm_mu_shift.append(v)
    
#%%
#create 10K 100dim vectors ~N(0,1) first 50 dims and ~N(-1,1) for last 50 dims

norm_zero = []
for i in range(10000):
    v1 = np.random.normal(0,1,10)
    v2 = np.array([0]*90)
    v = np.concatenate((v1,v2))
    norm_zero.append(v)
    
#%%
#create 10K 100dim vectors switch between ~N(0,1) and ~N(-1,1) every 10 dims

norm_shift_cluster = []
for i in range(10000):
    v = np.array([])
    for j in range(5):
        v1 = np.random.normal(0,1,10)
        v2 = np.random.normal(-1,1,10)
        v = np.concatenate((v,v1,v2))
    norm_shift_cluster.append(v)
 
#%%
#create 10K 100dim vectors switch between ~N(0,1) and ~N(0,5) every 10 dims

norm_spread_cluster = []
for i in range(10000):
    v = np.array([])
    for j in range(5):
        v1 = np.random.normal(0,1,10)
        v2 = np.random.normal(0,5,10)
        v = np.concatenate((v,v1,v2))
    norm_spread_cluster.append(v)
#%%
#create 10K 100dim vectors ~exp(1)

exp_dist = []
for i in range(10000):
    v = np.random.exponential(size = 100)
    exp_dist.append(v)
    
#%%
#create 10K 100dim vectors ~exp(1) for first 50 dims ~exp(10) for last 50 dims

exp_50 = []
for i in range(10000):
    v1 = np.random.exponential(scale =1,size = 50)
    v2 = np.random.exponential(scale = 10, size = 50)
    v = np.concatenate((v1,v2))
    exp_50.append(v)
    
#%%
#create 10K 100dim vectors ~exp(1) for first 50 dims 0 for last 50 dims

exp_zero = []
for i in range(10000):
    v1 = np.random.exponential(scale =1,size = 50)
    v2 = np.array([0]*50)
    v = np.concatenate((v1,v2))
    exp_zero.append(v)

#%%    
uni_norm = []
for i in range(10000):
    v1 = np.random.uniform(-1,1,50)
    v2 = np.random.normal(0,1,50)
    v = np.concatenate((v1,v2))
    uni_norm.append(v)
    
#%%
uni_exp = []
for i in range(10000):
    v1 = np.random.uniform(-1,1,50)
    v2 = np.random.exponential(scale = 1, size = 50)
    v = np.concatenate((v1,v2))
    uni_exp.append(v)
    
#%%    
norm_exp = []
for i in range(10000):
    v1 = np.random.exponential(scale=1, size = 50)
    v2 = np.random.normal(0,1,50)
    v = np.concatenate((v1,v2))
    norm_exp.append(v)