#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:14:35 2022

@author: anna
"""
import numpy as np
from sklearn.decomposition import PCA
from math import exp
import matplotlib.pyplot as plt
import seaborn as sns
from random import random

#%%
models = {"uni": uni, 
          "uni_75":uni_75, 
          "uni_50":uni_50, 
          "uni_25":uni_25, 
          "uni_neg":uni_neg, 
          "uni_spread_cluster":uni_spread_cluster, 
          "uni_shift_cluster":uni_shift_cluster,
          "uni_zero_cluster":uni_zero_cluster, 
          "uni_50_zero":uni_50_zero, 
          "uni_opp":uni_opp, 
          "norm":norm, 
          "norm_sd_spread":norm_sd_spread, 
          "norm_mu_shift":norm_mu_shift, 
          "norm_zero":norm_zero, 
          "norm_spread_cluster":norm_spread_cluster, 
          "norm_shift_cluster":norm_shift_cluster,
          "exp_dist":exp_dist,
          "exp_50":exp_50,
          "exp_zero":exp_zero,
          "uni_norm":uni_norm,
          "uni_exp":uni_exp,
          "norm_exp":norm_exp}

with open("disc_entropy_results.csv", "w") as f:
    for model in models:
        disc_ent = discrete_entropy_ratio(models[model],1000)
        row = ",".join([str(np.mean(disc_ent)), str(np.var(disc_ent))])+"\n"
        f.write(row)
        
from sklearn.decomposition import PCA

#%%
#1D example for computing Z
test_data_a = [[-1],[-1],[-1],[-1],[-1],[5]]

sums_1_a = 0
for emb in test_data_a:
    sums_1_a+=(exp(emb[0]))
    

test_data_b = [[1],[1],[1],[1],[1],[-5]]

sums_1_b = 0
for emb in test_data_b:
    sums_1_b+=(exp(emb[0]))
   
#%%
#3D example -- actually a little hard to visualize

test_3D_a = [[-1,0,0],[-1,0,0,],[-1,0,0],[-1,0,0],[-1,0,0],[5,0,0],
           [0,-1,0],[0,-1,0],[0,-1,0],[0,-1,0],[0,-1,0],[0,5,0],
           [0,0,-2],[0,0,-2],[0,0,-2],[0,0,-2],[0,0,-2],[0,0,10]]

test_3D_b = [[1,0,0],[1,0,0,],[1,0,0],[1,0,0],[1,0,0],[-5,0,0],
           [0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,-5,0],
           [0,0,2],[0,0,2],[0,0,2],[0,0,2],[0,0,2],[0,0,-10]]

sums_a = []
for c in range(len(test_3D_a[0])):
    c_sum = 0
    for emb in test_3D_a:
        c_sum+=(exp(emb[c]))
    sums_a.append(c_sum)
    
sums_b = []
for c in range(len(test_3D_b[0])):
    c_sum = 0
    for emb in test_3D_b:
        c_sum+=(exp(emb[c]))
    sums_b.append(c_sum)
    
import matplotlib.pyplot as plt    
fig = plt.figure()
ax = fig.add_subplot(121,projection = '3d')
ax.scatter(  jitter(np.array(test_3D_a).T[2],0),jitter(np.array(test_3D_a).T[1],0),jitter(np.array(test_3D_a).T[0],0),color = "blue")
ax2 = fig.add_subplot(122,projection = '3d')
ax2.scatter(  jitter(np.array(test_3D_b).T[2],0),jitter(np.array(test_3D_b).T[1],0),jitter(np.array(test_3D_b).T[0],0), color = "blue")
plt.show()

#%%
#2D orthogonal example

test_2D_a = [[-1,0],[-1,0],[-1,0],[-1,0],[-1,0],[5,0],
             [0,2],[0,2],[0,2],[0,2],[0,2],[0,-10]]

test_2D_b = [[1,0],[1,0],[1,0],[1,0],[1,0],[-5,0],
             [0,-2],[0,-2],[0,-2],[0,-2],[0,-2],[0,10]]

sums_a = []
for c in range(len(test_3D_a[0])):
    c_sum = 0
    for emb in test_3D_a:
        c_sum+=(exp(emb[c]))
    sums_a.append(c_sum)
    
Iw_a = min(sums_a)/max(sums_a)    
sums_b = []
for c in range(len(test_3D_b[0])):
    c_sum = 0
    for emb in test_3D_b:
        c_sum+=(exp(emb[c]))
    sums_b.append(c_sum)
    
Iw_b = min(sums_b)/max(sums_b)

import seaborn as sns
def jitter(values,j):
    return values + np.random.normal(j,0.1,values.shape)

f,axes = plt.subplots(1,2)
for ax in axes:
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.grid(True, which = 'both')

axes[0].text(1,-4,'I(w)=' +str(round(Iw_a,4)))
sns.scatterplot(jitter(np.array(test_2D_a).T[0],0), jitter(np.array(test_2D_a).T[1],0),color = 'blue', ax = axes[0])
axes[1].text(-3.5,4,'I(w)=' +str(round(Iw_b,4)))
sns.scatterplot(jitter(np.array(test_2D_b).T[0],0), jitter(np.array(test_2D_b).T[1],0),color = 'blue', ax = axes[1])

#%%
#2 dimensional using PCA and arbitrarily swapping sign of PCA components

np.random.seed(11)
rand_2D = []
for i in range(20):
    # x = np.random.choice(range(-10,10))
    # y = np.random.choice(range(-10,10))
    # rand_2D.append([x,y])
    rand_2D.append(np.random.normal(0,1,size = 2))
    
pc_2D = PCA()
pc_2D.fit(rand_2D)
pc_2D.components_
tx_2D_a = np.dot(rand_2D, pc_2D.components_.T)

neg_pc = pc_2D.components_ *(-1)
tx_2D_b = np.dot(rand_2D, neg_pc.T)

sums_a = []
for c in range(len(tx_2D_a[0])):
    c_sum = 0
    for emb in tx_2D_a:
        c_sum+=(exp(emb[c]))
    sums_a.append(c_sum)
    
Iw_a = min(sums_a)/max(sums_a)    

sums_b = []
for c in range(len(tx_2D_b[0])):
    c_sum = 0
    for emb in tx_2D_b:
        c_sum+=(exp(emb[c]))
    sums_b.append(c_sum)
    
Iw_b = min(sums_b)/max(sums_b)


f,axes = plt.subplots(1,2)
for ax in axes:
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.grid(True, which = 'both')

axes[0].text(0.4,-1.3,'I(V)=' +str(round(Iw_a,4)), fontsize = 'large')
sns.scatterplot(np.array(tx_2D_a).T[0], np.array(tx_2D_a).T[1],color = 'blue', ax = axes[0])
axes[1].text(-2.75,-1.35,'I(V)=' +str(round(Iw_b,4)), fontsize = 'large')
sns.scatterplot(np.array(tx_2D_b).T[0], np.array(tx_2D_b).T[1],color = 'blue', ax = axes[1])

#%%
#100 dimensional using PCA and arbitrarily swapping sign of PCA components
np.random.seed(11)
mu = np.random.choice(range(10), size = 100)
s = np.random.choice(range(30), size = 100)
rand_100D = []
for i in range(100):
    #m1 = np.random.choice(range(10)),
    #sd1 = np.random.choice(range(30))
    #sc2 = np.random.choice(range(30))
    emb = list(np.random.normal(mu[i],s[i],size = 100))
    #emb.extend(list(np.random.exponential(sc2,size = 50)))
    rand_100D.append(emb)
    
rand_100D = np.array(rand_100D).T

# ortho = np.zeros((1000,100))
# for i in range(100):
#     sign = np.random.choice([1])
#     for j in range(9):
#         scale = np.random.choice(range(30))
#         ortho[j+(i*10)][i] = sign*(scale)
#     ortho[((i+1)*10)-1][i] = -1*sign*(scale*9)
    
pc_100D = PCA()
pc_100D.fit(rand_100D)
pc_100D.components_
tx_100D_a = np.dot(rand_100D, pc_100D.components_.T)

neg_pc = pc_100D.components_ *(-1)
tx_100D_b = np.dot(rand_100D, neg_pc.T)

sums_a = []
for c in range(len(tx_100D_a[0])):
    c_sum = 0
    for emb in tx_100D_a:
        c_sum+=(exp(emb[c]/(np.linalg.norm(emb)+1e-10)))
    sums_a.append(c_sum)
    
Iw_a = min(sums_a)/max(sums_a)    

sums_b = []
for c in range(len(tx_100D_b[0])):
    c_sum = 0
    for emb in tx_100D_b:
        c_sum+=(exp(emb[c]/(np.linalg.norm(emb)+1e-10)))
    sums_b.append(c_sum)
    
Iw_b = min(sums_b)/max(sums_b)
                                                                                                                                                        
#%%
#2D symmetrical clusters
np.random.seed(11)
clust_x = []
clust_y = []
clust = []
for i in range(125):
    x = np.random.uniform(-0.15, 0.15, size = 1)[0]
    y = np.random.uniform(0.85,1.15, size = 1)[0]
    x1 = np.random.uniform(0.85,1.15, size = 1)[0]
    y1 = np.random.uniform(-0.15, 0.15, size = 1)[0]

    clust_x.extend([x,-x, x1, -x1])
    clust_y.extend([y,-y, y1, -y1])
    clust.append([x,y])
    clust.append([-x,-y])
    clust.append([x1,y1])
    clust.append([-x1,-y1])
    
clust = np.array(clust)    
cos_dist_clust = []
for i in range(len(clust)):
    for j in range(i,len(clust)):
        cos_dist_clust.append(np.dot(clust[i],clust[j])/((np.linalg.norm(clust[i]))*(np.linalg.norm(clust[j]))))

clust_cos = np.mean(cos_dist_clust)

f,ax = plt.subplots()
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.grid(True, which = 'both')

plt.scatter(clust_x, clust_y)
    

clust_pc = PCA()
tx_clust = clust_pc.fit_transform(clust)

sums_clust = []
for c in range(len(tx_clust[0])):
    c_sum = 0
    for emb in tx_clust:
        c_sum+=(exp(emb[c]/(np.linalg.norm(emb)+1e-10)))
    sums_clust.append(c_sum)
    
Iw_clust = min(sums_clust)/max(sums_clust)  

clust_pcr = PC_ratio(clust)
clust_auc = AUC_eigensum(clust)

clust_koleo = koleo_entropy(np.array(clust), 1)
clust_vas = vasicek_entropy(clust)
clust_kl = KL_divergence(clust)
clust_ovl = knn_overlap(clust, k= 10, plot = True)

#%%
#2D uneven clusters
np.random.seed(11)
clust_ue= np.random.uniform(-0.25, 0.25, size = (500,2))
clust_ue_k = []
for i in range(2):
    k = np.random.choice(range(250))
    clust_ue_k.append(k)
    for j in range(250):
        if j < k:
            clust_ue[j+(i*250)][i] = np.random.uniform(0.75, 1.25, size = 1)[0]
        else:
            clust_ue[j+(i*250)][i] = np.random.uniform(-0.75, -1.25, size = 1)[0]

    
clust_ue = np.array(clust_ue)    

f,ax = plt.subplots()
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.grid(True, which = 'both')

plt.scatter(clust_ue.T[0], clust_ue.T[1])

clust_ue_pc = PCA()
tx_clust_ue = clust_ue_pc.fit_transform(clust_ue)

sums_clust_ue = []
for c in range(len(tx_clust_ue[0])):
    c_sum = 0
    for emb in tx_clust_ue:
        c_sum+=(exp(emb[c]/(np.linalg.norm(emb)+1e-10)))
    sums_clust_ue.append(c_sum)
    
clust_ue_cos = avg_cos(clust_ue, num_sample = len(clust_ue))    
Iw_clust_ue = min(sums_clust_ue)/max(sums_clust_ue)  
clust_ue_pcr = PC_ratio(clust_ue)
clust_ue_auc = AUC_eigensum(clust_ue)
clust_ue_koleo = koleo_entropy(clust_ue, 1)
clust_ue_vas = vasicek_entropy(clust_ue)
clust_ue_kl = KL_divergence(clust_ue)
clust_ue_ovl = knn_overlap(clust_ue, k= 10, plot = True)

#%%
#2D shifted clusters
np.random.seed(11)
clust_s= np.random.uniform(-0.25, 0.25, size = (500,2))
clust_s_r = []
for i in range(2):
    r = np.random.uniform(0.5,1.5, size = 2)
    clust_s_r.append(r)
    for j in range(250):
        if j < 125:
            clust_s[j+(i*250)][i] = np.random.uniform(r[0]-0.25, r[0]+0.25, size = 1)[0]
        else:
            clust_s[j+(i*250)][i] = np.random.uniform(-r[1]-0.25, -r[1]+0.25, size = 1)[0]

    
clust_s = np.array(clust_s)    

f,ax = plt.subplots()
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.grid(True, which = 'both')

plt.scatter(clust_s.T[0], clust_s.T[1])

clust_s_pc = PCA()
tx_clust_s = clust_s_pc.fit_transform(clust_s)

sums_clust_s = []
for c in range(len(tx_clust_s[0])):
    c_sum = 0
    for emb in tx_clust_s:
        c_sum+=(exp(emb[c]/(np.linalg.norm(emb)+1e-10)))
    sums_clust_s.append(c_sum)
    
clust_s_cos = avg_cos(clust_s, num_sample = len(clust_s))    
Iw_clust_s = min(sums_clust_s)/max(sums_clust_s)  
clust_s_pcr = PC_ratio(clust_s)
clust_s_auc = AUC_eigensum(clust_s)
clust_s_koleo = koleo_entropy(clust_s, 1)
clust_s_vas = vasicek_entropy(clust_s)
clust_s_kl = KL_divergence(clust_s)
clust_s_ovl = knn_overlap(clust_s, k= 10, plot = True)
#%%
#unit circle (filled)
np.random.seed(11)
l_circ = np.sqrt(np.random.uniform(0,1, 500))
a_circ = np.pi*np.random.uniform(0,2, 500)

x_circ = l_circ*np.cos(a_circ)
y_circ = l_circ*np.sin(a_circ)

circ = list(zip(x_circ,y_circ))
cos_dist_circ = []
for i in range(len(circ)):
    for j in range(i,len(circ)):
        cos_dist_circ.append(np.dot(circ[i],circ[j])/((np.linalg.norm(circ[i]))*(np.linalg.norm(circ[j]))))

circ_cos = np.mean(cos_dist_circ)


f,ax = plt.subplots()
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.grid(True, which = 'both')

plt.scatter(x_circ, y_circ)

circ_pc = PCA()
tx_circ = circ_pc.fit_transform(circ)

sums_circ = []
for c in range(len(tx_circ[0])):
    c_sum = 0
    for emb in tx_circ:
        c_sum+=(exp(emb[c]/(np.linalg.norm(emb)+1e-10)))
    sums_circ.append(c_sum)
    
Iw_circ = min(sums_circ)/max(sums_circ)  
circ_pcr = PC_ratio(circ)
circ_auc = AUC_eigensum(circ)
circ_koleo = koleo_entropy(np.array(circ), 1)
circ_vas = vasicek_entropy(circ)
circ_kl = KL_divergence(np.array(circ))
circ_ovl = knn_overlap(circ, k= 10, plot = True)
#%%
#cone
np.random.seed(11)
l_cone = np.sqrt(np.random.uniform(0,1,500))
a_cone = np.pi*np.random.uniform(1/3,2/3, 500)


def ct(r, arr):
    a = np.concatenate((np.array([[2*np.pi]*len(arr)]).T, arr), axis = 1)
    si = np.sin(a)
    si.T[0] = 1
    si = np.cumprod(si, axis = 1)
    co = np.cos(a)
    co = np.roll(co, -1)
    return si*co*r

r = np.array([l_cone]).T
arr = np.array([a_cone]).T
cone = ct(r,arr)

f,ax = plt.subplots()
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.grid(True, which = 'both')

plt.scatter(cone.T[0], cone.T[1])

cone_pc = PCA()
tx_cone = cone_pc.fit_transform(cone)

sums_cone = []
for c in range(len(tx_cone[0])):
    c_sum = 0
    for emb in tx_cone:
        c_sum+=(exp(emb[c]/(np.linalg.norm(emb)+1e-10)))
    sums_cone.append(c_sum)

cone_cos = avg_cos(cone,num_sample = len(cone))    
Iw_cone = min(sums_cone)/max(sums_cone)  
cone_pcr = PC_ratio(cone)
cone_auc = AUC_eigensum(cone)
cone_koleo = koleo_entropy(np.array(cone), 1)
cone_vas = vasicek_entropy(cone)
cone_kl = KL_divergence(np.array(cone))
cone_ovl = knn_overlap(cone, k= 10, plot = True)

#%%
#unit circle (circumference)

np.random.seed(11)
a_line = np.pi*np.random.uniform(0,2, 500)

x_line = np.cos(a_line)
y_line = np.sin(a_line)

line = list(zip(x_line,y_line))
cos_dist_line = []
for i in range(len(line)):
    for j in range(i,len(line)):
        cos_dist_line.append(np.dot(line[i],line[j])/((np.linalg.norm(line[i]))*(np.linalg.norm(line[j]))))

line_cos = np.mean(cos_dist_line)

f,ax = plt.subplots()
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.grid(True, which = 'both')

plt.scatter(x_line, y_line)

line_pc = PCA()
tx_line = line_pc.fit_transform(line)

sums_line = []
for c in range(len(tx_line[0])):
    c_sum = 0
    for emb in tx_line:
        c_sum+=(exp(emb[c]/(np.linalg.norm(emb)+1e-10)))
    sums_line.append(c_sum)
    
Iw_line = min(sums_line)/max(sums_line)  
line_pcr = PC_ratio(line)
line_auc = AUC_eigensum(line)
line_koleo = koleo_entropy(np.array(line), 1)
line_vas = vasicek_entropy(line)
line_kl = KL_divergence(np.array(line))
line_ovl = knn_overlap(line, k= 10, plot = True)
#%%
#unit circle (rings)

np.random.seed(11)
a_ring = np.pi*np.random.uniform(0,2, 250)

x1_ring = np.cos(a_ring)
x2_ring = 0.5*np.cos(a_ring)
y1_ring = np.sin(a_ring)
y2_ring = 0.5*np.sin(a_ring)


ring = list(zip(x1_ring,y2_ring))
ring.extend(list(zip(x2_ring,y2_ring)))

cos_dist_ring = []
for i in range(len(ring)):
    for j in range(i,len(ring)):
        cos_dist_ring.append(np.dot(ring[i],ring[j])/((np.linalg.norm(ring[i]))*(np.linalg.norm(ring[j]))))

ring_cos = np.mean(cos_dist_ring)


f,ax = plt.subplots()
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.grid(True, which = 'both')

plt.scatter(np.concatenate((x1_ring,x2_ring)), np.concatenate((y1_ring,y2_ring)))

ring_pc = PCA()
tx_ring = ring_pc.fit_transform(ring)

sums_ring = []
for c in range(len(tx_ring[0])):
    c_sum = 0
    for emb in tx_ring:
        c_sum+=(exp(emb[c]/(np.linalg.norm(emb)+1e-10)))
    sums_ring.append(c_sum)
    
Iw_ring = min(sums_ring)/max(sums_ring)  
ring_pcr = PC_ratio(ring)
ring_auc = AUC_eigensum(ring)
ring_koleo = koleo_entropy(np.array(ring), 1)
ring_vas = vasicek_entropy(ring)
ring_kl = KL_divergence(np.array(ring))
ring_ovl = knn_overlap(ring, k= 10, plot = True)
#%%
f, axes = plt.subplots(3,2)
f.tight_layout()
for i in range(3):
    for ax in axes[i]:
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.grid(True, which = 'both')
        
sns.scatterplot(clust2.T[0], clust2.T[1] ax = axes[0,1])
axes[0,1].text(-0.5,-1.6,'Avg Cos=' +str(round(clust_cos,4)))
axes[0,1].text(-0.5,-2,'I(w)=' +str(round(Iw_clust,4)))

sns.scatterplot(sp2.T[0], sp2.T[1], ax = axes[0,0])
axes[0,0].text(-0.5,-1.4,'Avg Cos=' +str(round(circ_cos,4)))
axes[0,0].text(-0.5,-1.73,'I(w)=' +str(round(Iw_circ,4)))

sns.scatterplot(x_line, y_line, ax = axes[1,0])
axes[1,0].text(-0.5,-1.4,'Avg Cos=' +str(round(line_cos,4)))
axes[1,0].text(-0.5,-1.73,'I(w)=' +str(round(Iw_line,4)))

sns.scatterplot(np.concatenate((x1_ring,x2_ring)), np.concatenate((y1_ring,y2_ring)), ax = axes[1,1])
axes[1,1].text(-0.5,-1.4,'Avg Cos=' +str(round(ring_cos,4)))
axes[1,1].text(-0.5,-1.73,'I(w)=' +str(round(Iw_ring,4)))

sns.scatterplot(cone.T[0], cone.T[1], ax = axes[2,0])
axes[2,0].text(-0.25,-0.45,'Avg Cos=' +str(round(cone_cos,4)))
axes[2,0].text(-0.25,-0.6,'I(w)=' +str(round(Iw_cone,4)))


#%%
#100D symmetrical clusters
np.random.seed(11)
clust100 = np.random.uniform(-0.5, 0.5, size = (25000,100))
for i in range(100):
    for j in range(250):
        if j < 125:
            clust100[j+(i*250)][i] = np.random.uniform(0.5, 1.5, size = 1)[0]
        else:
            clust100[j+(i*250)][i] = np.random.uniform(-0.5, -1.5, size = 1)[0]

    
clust100 = np.array(clust100)    

clust100_pc = PCA()
tx_clust100 = clust100_pc.fit_transform(clust100)

sums_clust100 = []
for c in range(len(tx_clust100[0])):
    c_sum = 0
    for emb in tx_clust100:
        c_sum+=(exp(emb[c]/(np.linalg.norm(emb)+1e-10)))
    sums_clust100.append(c_sum)
    
clust100_cos = avg_cos(clust100, num_sample = 2500)    
Iw_clust100 = min(sums_clust100)/max(sums_clust100)  
clust100_pcr = PC_ratio(clust100)
clust100_auc = AUC_eigensum(clust100)
clust100_koleo = koleo_entropy(clust100, 1)
clust100_vas = vasicek_entropy(clust100)
clust100_kl = KL_divergence(clust100)
clust100_ovl = knn_overlap(clust100, k= 10, plot = True)
#%%
#100D unit sphere (filled)
np.random.seed(11)

u_circ100 = np.random.normal(0,1,(25000,100))
norm_circ100=np.array([np.sum(u_circ100**2, axis = 1) **(0.5)])
r_circ100 = np.array([np.random.uniform(0,1,size = 25000)**(1.0/100)])
circ100= r_circ100.T*u_circ100/norm_circ100.T


circ100_pc = PCA()
tx_circ100 = circ100_pc.fit_transform(circ100)

sums_circ100 = []
for c in range(len(tx_circ100[0])):
    c_sum = 0
    for emb in tx_circ100:
        c_sum+=(exp(emb[c]/(np.linalg.norm(emb)+1e-10)))
    sums_circ100.append(c_sum)
    
circ100_cos = avg_cos(circ100, num_sample = 2500)    
Iw_circ100 = min(sums_circ100)/max(sums_circ100)  
circ100_pcr = PC_ratio(circ100)
circ100_auc = AUC_eigensum(circ100, plot = True)
circ100_koleo = koleo_entropy(circ100, 1)
circ100_vas = vasicek_entropy(circ100)
circ100_kl = KL_divergence(circ100)
circ100_ovl = knn_overlap(circ100, k= 10, plot = True)


#%%
#unit circle (circumference)

np.random.seed(11)

u_line100 = np.random.normal(0,1,(25000,100))
norm_line100=np.array([np.sum(u_line100**2, axis = 1) **(0.5)])
line100= u_line100/norm_line100.T


line100_pc = PCA()
tx_line100 = line100_pc.fit_transform(line100)

sums_line100 = []
for c in range(len(tx_line100[0])):
    c_sum = 0
    for emb in tx_line100:
        c_sum+=(exp(emb[c]/(np.linalg.norm(emb)+1e-10)))
    sums_line100.append(c_sum)
    
line100_cos = avg_cos(line100, num_sample = 2500)    
Iw_line100 = min(sums_line100)/max(sums_line100)  
line100_pcr = PC_ratio(line100)
line100_auc = AUC_eigensum(line100)
line100_koleo = koleo_entropy(line100, 1)
line100_vas = vasicek_entropy(line100)
line100_kl = KL_divergence(line100)
line100_ovl = knn_overlap(line100, k= 10, plot = True)
#%%
#unit circle (rings)

np.random.seed(11)
u_ring100 = np.random.normal(0,1,(25000,100))
norm1_ring100=np.array([np.sum(u_ring100[:12500]**2, axis = 1) **(0.5)])
norm2_ring100=np.array([np.sum(u_ring100[12500:]**2, axis = 1) **(0.5)])
ring1_ring100= u_ring100[:12500]/norm1_ring100.T
ring2_ring100 = 0.5*u_ring100[12500:]/norm2_ring100.T
ring100 = np.concatenate([ring1_ring100,ring2_ring100])

ring100_pc = PCA()
tx_ring100 = ring100_pc.fit_transform(ring100)

sums_ring100 = []
for c in range(len(tx_ring100[0])):
    c_sum = 0
    for emb in tx_ring100:
        c_sum+=(exp(emb[c]/(np.linalg.norm(emb)+1e-10)))
    sums_ring100.append(c_sum)

ring100_cos = avg_cos(ring100, num_sample = 2500)    
Iw_ring100 = min(sums_ring100)/max(sums_ring100)  
ring100_pcr = PC_ratio(ring100)
ring100_auc = AUC_eigensum(ring100, plot = True)
ring100_koleo = koleo_entropy(ring100, 1)
ring100_vas = vasicek_entropy(ring100)
ring100_kl = KL_divergence(ring100)
ring100_ovl = knn_overlap(ring100, k= 10, plot = True)

#%% 100D cone
np.random.seed(11)
l_cone100 = np.sqrt(np.random.uniform(0,1, [25000,100]))
a_cone100 = np.pi*np.random.uniform(1/3,2/3, [25000,99])


def ct(r, arr):
    a = np.concatenate((np.array([[2*np.pi]*len(arr)]).T, arr), axis = 1)
    si = np.sin(a)
    si.T[0] = 1
    si = np.cumprod(si, axis = 1)
    co = np.cos(a)
    co = np.roll(co, -1)
    return si*co*r

r = l_cone100
arr = a_cone100
cone100 = ct(r,arr)

cone100_pc = PCA()
tx_cone100 = cone100_pc.fit_transform(cone100)

sums_cone100 = []
for c in range(len(tx_cone100[0])):
    c_sum = 0
    for emb in tx_cone100:
        c_sum+=(exp(emb[c]/(np.linalg.norm(emb)+1e-10)))
    sums_cone100.append(c_sum)

cone100_cos = avg_cos(cone100, num_sample = 2500)    
Iw_cone100 = min(sums_cone100)/max(sums_cone100)  
cone100_pcr = PC_ratio(cone100)
cone100_auc = AUC_eigensum(cone100, plot = True)
cone100_koleo = koleo_entropy(cone100, 1)
cone100_vas = vasicek_entropy(cone100)
cone100_kl = KL_divergence(cone100)
cone100_ovl = knn_overlap(cone100, k= 10, plot = True)

plt.hist(np.linalg.norm(circ100, axis = 1))

#%%
#100D uneven clusters
np.random.seed(11)
clust100_ue= np.random.uniform(-0.25, 0.25, size = (25000,100))
clust100_ue_k = []
for i in range(100):
    k = np.random.choice(range(250))
    clust100_ue_k.append(k)
    for j in range(250):
        if j < k:
            clust100_ue[j+(i*250)][i] = np.random.uniform(0.75, 1.25, size = 1)[0]
        else:
            clust100_ue[j+(i*250)][i] = np.random.uniform(-0.75, -1.25, size = 1)[0]

    
clust100_ue = np.array(clust100_ue)    

clust100_ue_pc = PCA()
tx_clust100_ue = clust100_ue_pc.fit_transform(clust100_ue)

sums_clust100_ue = []
for c in range(len(tx_clust100_ue[0])):
    c_sum = 0
    for emb in tx_clust100_ue:
        c_sum+=(exp(emb[c]/(np.linalg.norm(emb)+1e-10)))
    sums_clust100_ue.append(c_sum)
    
clust100_ue_cos = avg_cos(clust100_ue, num_sample = 2500)    
Iw_clust100_ue = min(sums_clust100_ue)/max(sums_clust100_ue)  
clust100_ue_pcr = PC_ratio(clust100_ue)
clust100_ue_auc = AUC_eigensum(clust100_ue)
clust100_ue_koleo = koleo_entropy(clust100_ue, 1)
clust100_ue_vas = vasicek_entropy(clust100_ue)
clust100_ue_kl = KL_divergence(clust100_ue)
clust100_ue_ovl = knn_overlap(clust100_ue, k= 10, plot = True)
#%%
#100D shifted clusters
np.random.seed(11)
clust100_s= np.random.uniform(-0.25, 0.25, size = (25000,100))

for i in range(100):
    r = np.random.uniform(0.5,1.5, size = 2)
    for j in range(250):
        if j < 125:
            clust100_s[j+(i*250)][i] = np.random.uniform(r[0]-0.25, r[0]+0.25, size = 1)[0]
        else:
            clust100_s[j+(i*250)][i] = np.random.uniform(-r[1]-0.25, -r[1]+0.25, size = 1)[0]

    
clust100_s = np.array(clust100_s)    

clust100_s_pc = PCA()
tx_clust100_s = clust100_s_pc.fit_transform(clust100_s)

sums_clust100_s = []
for c in range(len(tx_clust100_s[0])):
    c_sum = 0
    for emb in tx_clust100_s:
        c_sum+=(exp(emb[c]/(np.linalg.norm(emb)+1e-10)))
    sums_clust100_s.append(c_sum)
    
clust100_s_cos = avg_cos(clust100_s, num_sample = 2500)    
Iw_clust100_s = min(sums_clust100_s)/max(sums_clust100_s)  
clust100_s_pcr = PC_ratio(clust100_s)
clust100_s_auc = AUC_eigensum(clust100_s)
clust100_s_koleo = koleo_entropy(clust100_s, 1)
clust100_s_vas = vasicek_entropy(clust100_s)
clust100_s_kl = KL_divergence(clust100_s)
clust100_s_ovl = knn_overlap(clust100_s, k= 10, plot = True)

#%%
np.random.seed(11)
sp10 = sphere(10)
np.random.seed(11)
circ10 = sphere(10, fill = False)
np.random.seed(11)
ring10 = sphere(10,fill = False, rings = 2)
np.random.seed(11)
clust10 = cluster(10)
np.random.seed(11)
cone10 = sphere(10, cone = True)
np.random.seed(11)
clust10_ue = cluster(10,shuffle = True)
np.random.seed(11)
clust10_s = cluster(10,shift = True)

#%%
np.random.seed(11)
sp2 = sphere(2)
np.random.seed(11)
circ2 = sphere(2, fill = False)
np.random.seed(11)
ring2 = sphere(2,fill = False, rings = 2)
np.random.seed(11)
clust2 = cluster(2)
np.random.seed(11)
cone2 = sphere(2, cone = True)
np.random.seed(11)
clust2_ue = cluster(2,shuffle = True)
np.random.seed(11)
clust2_s = cluster(2,shift = True)

#%%
np.random.seed(11)
sp50 = sphere(50)
np.random.seed(11)
circ50 = sphere(50, fill = False)
np.random.seed(11)
ring50 = sphere(50,fill = False, rings = 2)
np.random.seed(11)
clust50 = cluster(50)
np.random.seed(11)
cone50 = sphere(50, cone = True)
np.random.seed(11)
clust50_ue = cluster(50,shuffle = True)
np.random.seed(11)
clust50_s = cluster(50,shift = True)

#%%
np.random.seed(11)
sp100 = sphere(100)
np.random.seed(11)
circ100 = sphere(100, fill = False)
np.random.seed(11)
ring100 = sphere(100,fill = False, rings = 2)
np.random.seed(11)
clust100 = cluster(100)
np.random.seed(11)
cone100 = sphere(100, cone = True)
np.random.seed(11)
clust100_ue = cluster(100,shuffle = True)
np.random.seed(11)
clust100_s = cluster(100,shift = True)

#%%
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.scatter(cone3.T[0], cone3.T[1], cone3.T[2])
plt.show()

#%%
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.scatter(sp3.T[0], sp3.T[1], sp3.T[2])
plt.show()

#%%
u_circ2 = np.random.normal(0,1,(500,2))
norm_circ2=np.array([np.linalg.norm(u_circ2, axis = 1)])
r_circ2 = np.array([np.random.uniform(0,1,size = 750)**(1.0/3)])

circ2= r_circ2.T*(u_circ2/norm_circ2.T)

min_u = np.min(u_circ2, axis = 0)
max_u = np.max(u_circ2, axis = 0)
u_cone2 = ((u_circ2-min_u))/(max_u-min_u) - np.array([0.5, 0])
norm_cone2=np.array([np.linalg.norm(u_cone2, axis = 1)])
cone2_cos = u_cone2/norm_cone2.T


theta = np.arccos(u_circ2/norm_circ2.T)
min_theta = np.min(theta, axis = 0)
max_theta = np.max(theta, axis = 0)
cone_theta = (np.pi/3)*(theta-min_theta)/(max_theta-min_theta)+(np.pi/3)
cone2_cos = np.cos(cone_theta)

cone2= r_circ2.T*cone2_cos

plt.scatter(cone2.T[0], cone2.T[1])

cone2_cos = []
while len(cone2_cos) <750:
    u = np.random.normal(0,1,3)
    norm=np.linalg.norm(u)
    cos = u/norm.T
    if cos[0] < 0.5 and cos[0] > -0.5 and cos[1] < 0.5 and cos[1] > -0.5 and cos[2]< 1 and cos[2]> 0:
        cone2_cos.append(cos)

#%%
#100D normal/uniform sphere vector norm histogram vs. #10D normal/uniformsphere vector norm histogram
np.random.seed(11)
norm2 = np.random.normal(0,1,size = (500,2))
norm10 = np.random.normal(0,1,size = (2500,10))
norm100 = np.random.normal(0,1,size = (25000,100))

uniform2 = sphere(2)
uniform10 = sphere(10)
uniform100 = sphere(100)

# v2 = np.var(uniform2)
# v10 = np.var(uniform10)
# v100 = np.var(uniform100)

# norm2 = np.random.normal(0,v2,size = (500,2))
# norm10 = np.random.normal(0,v10,size = (2500,10))
# norm100 = np.random.normal(0,v100,size = (25000,100))

# clust2 = cluster(2)
# clust10 = cluster(10)
# clust100 = cluster(100)

cone2 = sphere(2, cone =True)
cone10 = sphere(10, cone = True)
cone100 = sphere(100, cone = True)

#transform to have same variance as normal

uniform2 *= np.sqrt(2/sum(np.var(uniform2,axis = 0)))
uniform10 *= np.sqrt(10/sum(np.var(uniform10,axis = 0)))
uniform100 *= np.sqrt(100/sum(np.var(uniform100,axis = 0)))
# clust2 *= np.sqrt(2/sum(np.var(clust2,axis = 0)))
# clust10 *= np.sqrt(10/sum(np.var(clust10,axis = 0)))
# clust100 *= np.sqrt(100/sum(np.var(clust100,axis = 0)))
cone2 *= np.sqrt(2/sum(np.var(cone2,axis = 0)))
cone10 *= np.sqrt(10/sum(np.var(cone10,axis = 0)))
cone100 *= np.sqrt(100/sum(np.var(cone100,axis = 0)))
#%%

fig,axes = plt.subplots(1,3, figsize = (12,4))

fig.suptitle('Sphere', fontsize = 24)
fig.tight_layout(rect = (0.025,0.1,1,1),pad = 1.9)
axes[0].set_title('Dim = 2', fontsize = 18)
axes[0].hist(np.linalg.norm(uniform2, axis = 1), bins = 20, rwidth = 0.8)
axes[0].tick_params(labelsize = 16)
axes[0].set_xlabel('Vector Norm', fontsize = 18)
axes[1].set_title('Dim = 10', fontsize = 18)
axes[1].hist(np.linalg.norm(uniform10, axis = 1), bins = 20, rwidth = 0.8)
axes[1].tick_params(labelsize = 16)
axes[1].set_xlabel('Vector Norm', fontsize = 18)
axes[2].set_title('Dim = 100', fontsize = 18)
axes[2].hist(np.linalg.norm(uniform100, axis = 1), bins = 20, rwidth = 0.8)
axes[2].tick_params(labelsize = 16)
axes[2].set_xlabel('Vector Norm', fontsize = 18)

plt.savefig("sphere_norm.png")




#%%
fig = plt.figure(constrained_layout = True, figsize = (8,6))
(sf1,sf2,sf3,sf4) = fig.subfigures(4,1)
(ax1,ax2,ax3) = sf1.subplots(1,3)
(ax4,ax5,ax6) = sf2.subplots(1,3)
(ax7,ax8,ax9) = sf4.subplots(1,3)
(ax10,ax11,ax12) = sf3.subplots(1,3)

sf1.suptitle('Normal',fontsize = 12)
ax1.hist(np.linalg.norm(norm2, axis = 1))
ax1.tick_params(labelsize = 8)
ax1.set_title('Dim = 2', fontsize = 9)
ax2.hist(np.linalg.norm(norm10, axis = 1), bins = 20)
ax2.tick_params(labelsize = 8)
ax2.set_title('Dim = 10', fontsize = 9)
ax3.hist(np.linalg.norm(norm100, axis = 1), bins = 20)
ax3.tick_params(labelsize = 8)
ax3.set_title('Dim = 100', fontsize = 9)

sf2.suptitle('Sphere', fontsize = 12)
ax4.hist(np.linalg.norm(uniform2, axis = 1))
ax4.tick_params(labelsize = 8)
ax5.hist(np.linalg.norm(uniform10, axis = 1), bins = 20)
ax5.tick_params(labelsize = 8)
ax6.hist(np.linalg.norm(uniform100, axis = 1), bins = 20)
ax6.tick_params(labelsize = 8)

sf4.suptitle('Symmetric Clusters', fontsize = 12)
ax7.hist(np.linalg.norm(clust2, axis = 1))
ax7.tick_params(labelsize = 8)
ax7.set_xlabel('Vector Norm', fontsize = 9)
ax8.hist(np.linalg.norm(clust10, axis = 1), bins = 20)
ax8.tick_params(labelsize = 8)
ax8.set_xlabel('Vector Norm', fontsize = 9)
ax9.hist(np.linalg.norm(clust100, axis = 1), bins = 20)
ax9.tick_params(labelsize = 8)
ax9.set_xlabel('Vector Norm', fontsize = 9)

sf3.suptitle('Cone', fontsize = 12)
ax10.hist(np.linalg.norm(cone2, axis = 1))
ax10.tick_params(labelsize = 8)
#ax10.set_xlabel('Vector Norm')
ax11.hist(np.linalg.norm(cone10, axis = 1), bins = 20)
ax11.tick_params(labelsize = 8)
#ax11.set_xlabel('Vector Norm')
ax12.hist(np.linalg.norm(cone100, axis = 1), bins = 20)
ax12.tick_params(labelsize = 8)
#ax12.set_xlabel('Vector Norm')
plt.show()

#%%
#10D normal/sphere/cone knn dist vs 100D normal/sphere/cone knn dist

np.random.seed(11)
# norm10 = np.random.normal(0,1,size = (2500,10))
# norm100 = np.random.normal(0,1,size = (25000,100))

uniform2 = sphere(2)
uniform10 = sphere(10)
uniform100 = sphere(100)

nest2 = sphere(2, rings = 2, fill = False)
nest10 = sphere(10, rings = 2, fill = False)
nest100 = sphere(100, rings = 2, fill = False)

# v10 = np.var(uniform10)
# v100 = np.var(uniform100)

# norm10 = np.random.normal(0,v10,size = (2500,10))
# norm100 = np.random.normal(0,v100,size = (25000,100))

unicone2 = sphere(2, cone = True)
unicone10 = sphere(10, cone = True)
unicone100 = sphere(100,cone = True)

clust2 = cluster(2)
clust10 = cluster(10)
clust100 = cluster(100)

#transform to have same variance as normal
uniform2 *= np.sqrt(2/sum(np.var(uniform2,axis = 0)))
uniform10 *= np.sqrt(10/sum(np.var(uniform10,axis = 0)))
uniform100 *= np.sqrt(100/sum(np.var(uniform100,axis = 0)))
unicone2 *= np.sqrt(2/sum(np.var(unicone2,axis = 0)))
unicone10 *= np.sqrt(10/sum(np.var(unicone10,axis = 0)))
unicone100 *= np.sqrt(100/sum(np.var(unicone100,axis = 0)))
clust2 *= np.sqrt(2/sum(np.var(clust2,axis = 0)))
clust10 *= np.sqrt(10/sum(np.var(clust10,axis = 0)))
clust100 *= np.sqrt(100/sum(np.var(clust100,axis = 0)))
nest2 *= np.sqrt(2/sum(np.var(nest2,axis = 0)))
nest10 *= np.sqrt(10/sum(np.var(nest10,axis = 0)))
nest100 *= np.sqrt(100/sum(np.var(nest100,axis = 0)))


#compute nn dist for each point in each dist

index_n2 = faiss.IndexFlatL2(2)
index_n2.add(np.array(nest2).astype('float32'))
D_n2,I_n2 = index_n2.search(np.array(nest2).astype('float32'), 2)
k_n2 = np.sqrt(np.array([i[1] for i in D_n2]))

index_n10 = faiss.IndexFlatL2(10)
index_n10.add(np.array(nest10).astype('float32'))
D_n10,I_n10 = index_n10.search(np.array(nest10).astype('float32'), 2)
k_n10 = np.sqrt(np.array([i[1] for i in D_n10]))

index_n100 = faiss.IndexFlatL2(100)
index_n100.add(np.array(nest100).astype('float32'))
D_n100,I_n100 = index_n100.search(np.array(nest100).astype('float32'), 2)
k_n100 = np.sqrt(np.array([i[1] for i in D_n100]))

index_u2 = faiss.IndexFlatL2(2)
index_u2.add(np.array(uniform2).astype('float32'))
D_u2,I_u2 = index_u2.search(np.array(uniform2).astype('float32'), 2)
k_u2 = np.sqrt(np.array([i[1] for i in D_u2]))

index_u10 = faiss.IndexFlatL2(10)
index_u10.add(np.array(uniform10).astype('float32'))
D_u10,I_u10 = index_u10.search(np.array(uniform10).astype('float32'), 2)
k_u10 = np.sqrt(np.array([i[1] for i in D_u10]))

index_u100 = faiss.IndexFlatL2(100)
index_u100.add(np.array(uniform100).astype('float32'))
D_u100,I_u100 = index_u100.search(np.array(uniform100).astype('float32'), 2)
k_u100 = np.sqrt(np.array([i[1] for i in D_u100]))

index_c2 = faiss.IndexFlatL2(2)
index_c2.add(np.array(unicone2).astype('float32'))
D_c2,I_c2 = index_c2.search(np.array(unicone2).astype('float32'), 2)
k_c2 = np.sqrt(np.array([i[1] for i in D_c2]))

index_c10 = faiss.IndexFlatL2(10)
index_c10.add(np.array(unicone10).astype('float32'))
D_c10,I_c10 = index_c10.search(np.array(unicone10).astype('float32'), 2)
k_c10 = np.sqrt(np.array([i[1] for i in D_c10]))

index_c100 = faiss.IndexFlatL2(100)
index_c100.add(np.array(unicone100).astype('float32'))
D_c100,I_c100 = index_c100.search(np.array(unicone100).astype('float32'), 2)
k_c100 = np.sqrt(np.array([i[1] for i in D_c100]))

index_cl2 = faiss.IndexFlatL2(2)
index_cl2.add(np.array(clust2).astype('float32'))
D_cl2,I_cl2 = index_cl2.search(np.array(clust2).astype('float32'), 2)
k_cl2 = np.sqrt(np.array([i[1] for i in D_cl2]))

index_cl10 = faiss.IndexFlatL2(10)
index_cl10.add(np.array(clust10).astype('float32'))
D_cl10,I_cl10 = index_cl10.search(np.array(clust10).astype('float32'), 2)
k_cl10 = np.sqrt(np.array([i[1] for i in D_cl10]))

index_cl100 = faiss.IndexFlatL2(100)
index_cl100.add(np.array(clust100).astype('float32'))
D_cl100,I_cl100 = index_cl100.search(np.array(clust100).astype('float32'), 2)
k_cl100 = np.sqrt(np.array([i[1] for i in D_cl100]))

#%%
#visualize knn distance 
fig = plt.figure(constrained_layout = True,figsize = (8,6))
(sf1,sf2,sf3,sf4) = fig.subfigures(4,1)
(ax1,ax2,ax3) = sf1.subplots(1,3)
(ax4,ax5,ax6) = sf2.subplots(1,3)
(ax7,ax8,ax9) = sf3.subplots(1,3)
(ax10,ax11,ax12) = sf4.subplots(1,3)

sf2.suptitle('Nested Shell', fontsize = 12)
ax4.hist(k_n2, bins = 20)
ax4.tick_params(labelsize = 8)
ax5.hist(k_n10, bins = 20)
ax5.tick_params(labelsize = 8)
ax6.hist(k_n100, bins = 20)
ax6.tick_params(labelsize = 8)

sf1.suptitle('Sphere', fontsize = 12)
ax1.set_title('Dim = 2', fontsize = 9)
ax1.hist(k_u2, bins = 20)
ax1.tick_params(labelsize = 8)
ax2.hist(k_u2, bins = 20)
ax2.set_title('Dim = 10', fontsize = 9)
ax2.tick_params(labelsize = 8)
ax3.hist(k_u100, bins = 20)
ax3.set_title('Dim = 100', fontsize = 9)
ax3.tick_params(labelsize = 8)

sf3.suptitle('Cone', fontsize = 12)
ax7.hist(k_c2, bins = 20)
ax7.tick_params(labelsize = 8)
ax8.hist(k_c10, bins = 20)
ax8.tick_params(labelsize = 8)
ax9.hist(k_c100, bins = 20)
ax9.tick_params(labelsize = 8)

sf4.suptitle('Symmetric Clusters', fontsize = 12)
ax10.hist(k_cl2, bins = 20)
ax10.tick_params(labelsize = 8)
ax10.set_xlabel('Nearest Neighbor Distance', fontsize = 9)
ax11.hist(k_cl10, bins = 20)
ax11.tick_params(labelsize = 8)
ax11.set_xlabel('Nearest Neighbor Distance', fontsize = 9)
ax12.hist(k_cl100, bins = 20)
ax12.tick_params(labelsize = 8)
ax12.set_xlabel('Nearest Neighbor Distance', fontsize = 9)
plt.show()

#%%    
knn_overlap(cone10, plot = True)
#%%
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
        r4 = np.random.uniform(0,r2[0],size = (dim*250))
        r5 = r4**(1.0/(dim-1))
        u = np.random.normal(0,1,(dim*250,dim-1))
        norm = np.array([np.linalg.norm(u, axis = 1)])
        sp = r3.T*u/norm.T
        c = np.concatenate((r.T, sp), axis = 1)
        
    else:
        u = np.random.normal(0,1,(dim*250,dim))
        norm = np.array([np.linalg.norm(u, axis = 1)])
        c = r.T*u/norm.T    

    return c

#%%
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.scatter(c.T[0], c.T[1], c.T[2])
plt.show()
#%%
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.scatter(cone3.T[0], cone3.T[1], cone3.T[2])
plt.show()

#%%
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.scatter(sp3.T[0], sp3.T[1], sp3.T[2])
plt.show()

#%%

l = [0]
l.extend([100]*99)
l = np.array(l)
ref = np.cumsum([1]*100)
c = [np.log(i)*(100/np.log(100)) for i in range(1,101)]

plt.title("Eigenvalue Cumulative Sum")
plt.xlabel("Principal Component")
plt.plot(ref, l, label = 'One dimension')
plt.plot(ref, c, label = 'Limited dimensions')
plt.plot(ref,ref, label = 'All dimensions')
plt.legend()