#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:39:16 2022

@author: anna
"""

from sklearn.decomposition import PCA
from sklearn.metrics import auc
from scipy.stats import entropy, differential_entropy, multivariate_normal
from scipy.special import gamma, kl_div, rel_entr
from scipy.spatial import KDTree
import numpy as np
from math import exp,e
import argparse
import matplotlib.pyplot as plt  
import faiss 
from sympy import EulerGamma

def discrete_entropy_ratio(embeddings, k):
    #transform embeddings into full-rank PCA projections
    pc = PCA()
    px_embeddings = pc.fit_transform(embeddings)
    min_v = np.min(px_embeddings)
    max_v = np.max(px_embeddings)
    
    #for each dim (pc), separate into k bins and calculate frequency of projected embeddings falling into each bin
    dims = px_embeddings.T
    dim_freq = []
    for dim in dims:
        
        #use np.histogram to bin data in k equal bins
        hist, edges = np.histogram(dim, bins = k, range = (min_v, max_v), density = False)
        
        #normalize by total number embeddings
        dim_freq.append(hist/len(px_embeddings))

    #calculate entropy over each dimension and divide by maximum entropy given k bins
    ents = []
    max_ent = entropy([1/k]*k)
    for dim in dim_freq:
        ents.append(entropy(dim)/max_ent)
        
    return ents
            
def PC_ratio(embeddings):
    pc = PCA()
    pc.fit(embeddings)
    r = pc.explained_variance_[-1]/pc.explained_variance_[0]
    return round(r,4)

def AUC_eigensum(embeddings, plot = False):
    pc = PCA()
    pc.fit(embeddings)
    num_pc = pc.n_components_
    eigensum = np.cumsum(pc.explained_variance_)
    ref = np.cumsum([eigensum[-1]/num_pc]*num_pc)
    AUC_sum = eigensum-ref
    AUC = auc(range(num_pc),AUC_sum)
    
    total_poss = (eigensum[-1]*num_pc)/2
    
    if plot == True:
        plt.plot(range(num_pc), eigensum, label = 'Cumulative Sum of Eigenvalues')
        plt.plot(range(num_pc), ref)
        plt.legend()
        plt.show()
        
    return round(AUC/total_poss,4)

def normal_compare(dimension, size):
    x = multivariate_normal([0]*dimension, np.identity(dimension))
    y = x.rvs(size = size)
    pc = PCA()
    v = sum(pc.fit(y).explained_variance_)
    return x,y,v

#to recover total explained variance in component space w/ scaling factor--np.sqrt(scale)
#np.sum(np.cov((pc.transform(y)*np.sqrt(2)).T))    

def vasicek_entropy(embeddings):
    N = len(embeddings)
    d = len(embeddings[0])
    pc = PCA()
    tx = pc.fit_transform(embeddings)
    
    #fix total explained variance in data to match that of a standard normal distribution
    comp,sample, ev = normal_compare(d,N)
    scale = ev/sum(pc.explained_variance_)
    tx *= np.sqrt(scale)
     
    #compute theoretical max vasicek entropy for std normal comparison
    m = np.log(np.sqrt(2*np.pi*e))
    
    ent = differential_entropy(tx)
    r = np.exp(ent)/np.exp(m)
    
    #report mean SSE from ratio of 1
    return round(np.sum((1-r)**2)/d,4)

def koleo_entropy(embeddings, k, plot = False):
    N = len(embeddings)
    m = len(embeddings[0])
    
    #fix total variance in embeddings to match that of a standard normal
    tx = embeddings* np.sqrt(m/sum(np.var(embeddings,axis = 0)))
    
    # #compute k nearest neigbors for observed data
    index = faiss.IndexFlatL2(m)
    index.add(np.array(tx).astype('float32'))
    D,I = index.search(np.array(tx).astype('float32'), k+1)
    
    # #for kth nearest neighbor, compute koleo entropy using formula from Beirlant
    rho =np.sqrt(np.array([D[i][-1] for i in range(N)]))
    # tx_tree = KDTree(tx)
    # rho,I = tx_tree.query(tx, k = k+1)
    # rho = rho[:,k]
    H_obs = (1/N)*np.sum(np.log((N*rho)+1e-16))+np.log(2)+ EulerGamma.evalf()
    
    # #compare to koleo entropy for simulated std normal
    comp,sample,ev = normal_compare(m,N)
    index_n = faiss.IndexFlatL2(m)
    index_n.add(np.array(sample).astype('float32'))
    D_n,I_n = index_n.search(np.array(sample).astype('float32'), k+1)
    rho_n = np.sqrt(np.array([D_n[i][-1] for i in range(N)]))
    # n_tree = KDTree(sample)
    # rho_n,I_n = n_tree.query(sample, k = k+1)
    # rho_n = rho_n[:,k]
    H_n = (1/N)*np.sum(np.log(N*rho_n))+np.log(2)+ EulerGamma.evalf()
    
    
    if plot == True:
        plt.hist(rho,bins = 30, label = 'Observed Nearest Neighbors', histtype = 'step')
        plt.hist(rho_n,bins = 30, label = 'Normal Nearest Neighbors', histtype = 'step')
        plt.legend()
        plt.show()
    
    return round(H_obs/H_n,4)

def knn_overlap(embeddings, plot = False):
    N = len(embeddings)
    m = len(embeddings[0])
    k = min(m*5, 100)
    
    #compute k nearest neigbors for observed data
    index = faiss.IndexFlatL2(m)
    index.add(np.array(embeddings).astype('float32'))
    D,I = index.search(np.array(embeddings).astype('float32'), k+1)
    
    # #find first and kth nearest neighbor
    rho_1 = np.sqrt(np.array([D[i][1] for i in range(N)]))
    rho_k = np.sqrt(np.array([D[i][-1] for i in range(N)]))
    
    # rho_tree = KDTree(embeddings)
    # rho_k, I = rho_tree.query(embeddings, k = k+1)
    # rho_1 = rho_k[:,1]
    # rho_k = rho_k[:,k]
    
    rho_min = np.min([np.min(rho_1), np.min(rho_k)])
    rho_max = np.max([np.max(rho_1), np.max(rho_k)])
    
    if plot == True:
        plt.hist(rho_1,bins = 30,range=(rho_min, rho_max), label = 'First Nearest Neighbors', histtype = 'step')
        plt.hist(rho_k,bins = 30,range=(rho_min, rho_max), label = f'{k}th Nearest Neighbors', histtype = 'step')
        plt.legend()
        plt.show()
    
    rho_1_bin = np.histogram(rho_1, bins = 30, range=(rho_min, rho_max))
    rho_k_bin = np.histogram(rho_k, bins = 30, range = (rho_min, rho_max))
    rho_difference = rho_k_bin[0] - rho_1_bin[0]
    rho_ovl =  rho_k_bin[0]-[i if i >0 else 0 for i in rho_difference]
    
    ovl_perc = np.sum(rho_ovl)/np.sum(rho_k_bin[0])
    
    return round(ovl_perc,4)
    
def KL_divergence(embeddings):
    N = len(embeddings)
    d = len(embeddings[0])
    pc = PCA()
    tx = pc.fit_transform(embeddings)
    
    #fix total explained variance in data to match that of a standard normal distribution
    comp,sample,ev = normal_compare(d,N)
    scale = ev/sum(pc.explained_variance_)
    tx *= np.sqrt(scale)
    m_obs = np.mean(tx, axis = 0)
    s_obs = np.cov(tx.T)  
    
    #compute KL divergence w/ closed form for two gaussians
    kl = 0.5*((m_obs@m_obs)+np.trace(s_obs)-d-np.log(np.linalg.det(s_obs)+1e-16))
    
    return round(kl,4)

#only valid for univariate data
#https://medium.com/datalab-log/measuring-the-statistical-similarity-between-two-samples-using-jensen-shannon-and-kullback-leibler-8d05af514b15    

# def ecdf(embeddings): 
#     embeddings = np.sort(embeddings)
#     u, c = np.unique(embeddings, return_counts=True)
#     n = len(embeddings)
#     y = (np.cumsum(c) - 0.5)/n
#     def interpolate_(embeddings_):
#         yinterp = np.interp(embeddings_, u, y, left=0.0, right=1.0)
#         return yinterp
#     return interpolate_

# def kl_divergence_empirical(embeddings):
#     N = len(embeddings)
#     d = len(embeddings[0])
    
#     tx = embeddings * np.sqrt(d/sum(np.var(embeddings,axis = 0)))
#     comp, sample, ev = normal_compare(d,N)
    
#     d_emb = np.diff(np.sort(np.unique(embeddings)))
#     d_comp = np.diff(np.sort(np.unique(sample)))
#     e_emb = np.min(d_emb)
#     e_comp = np.min(d_comp)
#     e = np.min([e_emb,e_comp])*0.5
#     P = ecdf(embeddings)
#     Q = ecdf(sample)  
#     KL = (1./N)*np.sum(np.log((P(embeddings) - P(embeddings-e))/(Q(embeddings) - Q(embeddings-e))))
    
#     return KL

def kl_divergence_discrete(embeddings):
    N = len(embeddings)
    d = len(embeddings[0])
    pc = PCA()
    tx = pc.fit_transform(embeddings)
    
    #fix total explained variance in data to match that of a standard normal distribution
    comp,sample,ev = normal_compare(d,N)
    scale = ev/sum(pc.explained_variance_)
    tx *= np.sqrt(scale)
    
    #bin data per dimension from normal sample and create reference prob dist
    norm_min = np.min(sample)
    norm_max = np.max(sample)
    
    norm_bin = np.array([np.histogram(sample.T[i], bins = 30, range = (norm_min, norm_max))[0]+1e-16 for i in range(d)])
    norm_p = norm_bin/np.array([np.sum(norm_bin, axis = 1)]).T
    
    #use same bins on ea. component of observed data to create discrete obs prob dist
    tx_bin = np.array([np.histogram(tx.T[i], bins = 30, range = (norm_min, norm_max))[0]+1e-16 for i in range(d)])
    tx_p = tx_bin/np.array([np.sum(tx_bin, axis = 1)]).T
    
    #compute kl-divergence per dimension/component
    kl = np.array([entropy(tx_p[i], norm_p[i]) for i in range(d)])
    
    #report mean SSE from divergence of 0
    return round(np.sum(kl**2)/d,4)

def kl_divergence_empirical(embeddings, k):
    N = len(embeddings)
    d = len(embeddings[0])
    pc = PCA()
    tx = pc.fit_transform(embeddings)
    
    comp,sample,ev = normal_compare(d,N)
      
    #fix total explained variance in data to match that of a standard normal distribution
    scale = ev/sum(pc.explained_variance_)
    tx *= np.sqrt(scale)
    tx = tx.copy(order = 'C')
    
    #find distances to kth nearest neighbor in observed data (r_k)
    # r_tree = KDTree(tx)
    # r_k, I = r_tree.query(tx, k = k+1)
    # r_k = r_k[:,k]
    
    # s_tree = KDTree(sample)
    # s_k, I = s_tree.query(tx, k = k+1)
    # s_k = s_k[:,k]
    
    #compute k nearest neigbor distance for observed data
    index = faiss.IndexFlatL2(d)
    index.add(np.array(tx).astype('float32'))
    D,I = index.search(np.array(tx).astype('float32'), k+1)
    r_k = np.sqrt(np.array([D[i][-1] for i in range(N)]))
    
    #compute k nearest neighbor distance in reference data
    comp,sample,ev = normal_compare(d,N)
    index_n = faiss.IndexFlatL2(d)
    index_n.add(np.array(sample).astype('float32'))
    D_n,I_n = index_n.search(np.array(tx).astype('float32'), k)
    s_k = np.sqrt(np.array([D_n[i][-1] for i in range(N)]))
    
    return round((d/N)*np.sum(np.max([[0]*N,np.log((s_k+1e-16)/(r_k+1e-16))], axis = 0)),4)
    
    
def isotropy_measures(embeddings, cos_samples, seed = 11):
    np.random.seed(seed)
    cos = avg_cos(embeddings, num_sample = cos_samples)
    iw = Iw(embeddings)
    pcr = PC_ratio(embeddings)
    auc = AUC_eigensum(embeddings)
    vas = vasicek_entropy(embeddings)
    koleo = koleo_entropy(embeddings, 1)
    ovl = knn_overlap(embeddings, 10)
    kl = KL_divergence(embeddings)
    kl_disc = kl_divergence_discrete(embeddings)
    kl_emp = kl_divergence_empirical(embeddings, 1)
    
    return cos,iw,pcr,auc,vas,koleo,ovl,kl,kl_disc,kl_emp