#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply isotropy measures to word2vec embeddings with jitter/clustering perturbations.

Designed for cell-by-cell execution in Spyder or VS Code interactive mode.
Requires alternate_isotropy_measures.py to be loaded in the session.
"""

import gensim.downloader
import numpy as np
from sklearn.cluster import KMeans
import math

#%%
w2v = gensim.downloader.load('word2vec-google-news-300')

#%%
#calc two top measures for w2v
w2v_v = np.array([w2v[i] for i in np.random.randint(0,len(w2v), 75000)])
vasicek_entropy(w2v_v)
AUC_eigensum(w2v_v)



#%%
#add gradient of uniform noise and recompute
jitter = [0,0.0001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1, 1.5, 3, 5]
jittered_w2v = []
jittered_scores = []

for j in jitter:
    print(j)
    randj = np.random.uniform(-j,j,[75000,300])
    w2v_j = w2v_v +randj
    jittered_w2v.append(w2v_j)
    vas_j = vasicek_entropy(w2v_j)
    auc_j = AUC_eigensum(w2v_j)
    cos = avg_cos(w2v_j)
    iw = Iw(w2v_j)
    jittered_scores.append((vas,auc))
    print("vas = ", vas_j)
    print("auc = ", auc_j)
    print("cos = ", cos)
    print("iw = ", iw)


#%%
#add gradient of normal noise and recompute
n_jittered_w2v = []
n_jittered_scores = []

for j in jitter:
    print(j)
    randj = np.random.normal(0,j,[75000,300])
    w2v_j = w2v_v +randj
    n_jittered_w2v.append(w2v_j)
    vas_j = vasicek_entropy(w2v_j)
    auc_j = AUC_eigensum(w2v_j)
    n_jittered_scores.append((vas,auc))
    print("vas = ", vas_j)
    print("auc = ", auc_j)

#%%
#fit kmeans centers
km = KMeans(n_clusters = 100).fit(w2v_v)

#calculate distances between w2v embeddings and closest centers
diff = []
for i in range(len(w2v_v)):
    c = km.labels_[i]
    c_loc = km.cluster_centers_[c]
    d = w2v_v[i]-c_loc
    diff.append(d)
    
#%%
#gradually move embeddings towards centers and record metrics

clustered_w2v = []
clustered_scores = []

for i in range(1,10):
    print(i)
    w2v_c = w2v_v + ((i/10)*np.array(diff))
    clustered_w2v.append(w2v_c)
    vas_c = vasicek_entropy(w2v_c)
    auc_c = AUC_eigensum(w2v_c)
    cos_c = avg_cos(w2v_c)
    iw_c = Iw(w2v_c)
    clustered_scores.append((vas_c,auc_c, cos_c, iw_c))
    print("vas = ", vas_c)
    print("auc = ", auc_c)
    print("cos = ", cos_c)
    print("iw = ", iw_c)
    
#%%
np.array(diff)*(1/10)
i =10
set([list(i) for i in clustered_w2v[9]])
