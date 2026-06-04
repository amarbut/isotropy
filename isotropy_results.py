#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 16:03:07 2022

@author: anna
"""
import pandas as pd
#%%
np.random.shuffle(embeddings)
samples = embeddings[0:num_sample]
cos_dist = []
for i in range(num_sample):
    for j in range(i,num_sample):
        cos_dist.append(np.dot(samples[i],samples[j])/((np.linalg.norm(samples[i])+1e-16)*(np.linalg.norm(samples[j])+1e-16)))
        
plt.hist(cos_dist, bins = 50)


np.random.seed(111)
cone2 = sphere(2,cone_start = 5/12, cone_stop = 7/12)
cone10 = sphere(10,cone_start = 5/12, cone_stop = 7/12)
cone50 = sphere(50,cone_start = 5/12, cone_stop = 7/12)
cone100 = sphere(100,cone_start = 5/12, cone_stop = 7/12)

#%%

np.random.seed(11)
sp = sphere(dim)
sprow = "sp"+str(dim)+","+",".join([str(i).replace('(', '').replace(')','') for i in isotropy_measures(sp, cos_samples = np.min([dim*250, 2500]))])+"\n"

np.random.seed(11)
circ = sphere(dim, fill = False)
circrow = "circ"+str(dim)+","+",".join([str(i).replace('(', '').replace(')','') for i in isotropy_measures(circ, cos_samples = np.min([dim*250, 2500]))])+"\n"

np.random.seed(11)
ring = sphere(dim, fill = False, rings = 2)
ringrow = "ring"+str(dim)+","+",".join([str(i).replace('(', '').replace(')','') for i in isotropy_measures(ring, cos_samples = np.min([dim*250, 2500]))])+"\n"

np.random.seed(11)
clust = cluster(dim)
clustrow = "clust"+str(dim)+","+",".join([str(i).replace('(', '').replace(')','') for i in isotropy_measures(clust, cos_samples = np.min([dim*250, 2500]))])+"\n"

np.random.seed(11)
cone = sphere(dim, cone= True)
conerow = "cone"+str(dim)+","+",".join([str(i).replace('(', '').replace(')','') for i in isotropy_measures(cone, cos_samples = np.min([dim*250, 2500]))])+"\n"

np.random.seed(11)
clust_ue = cluster(dim, shuffle = True)
clust_uerow = "clust_ue"+str(dim)+","+",".join([str(i).replace('(', '').replace(')','') for i in isotropy_measures(clust_ue, cos_samples = np.min([dim*250, 2500]))])+"\n"

np.random.seed(11)
clust_s = cluster(dim, shift = True)
clust_srow = "clust_s"+str(dim)+","+",".join([str(i).replace('(', '').replace(')','') for i in isotropy_measures(clust_s, cos_samples = np.min([dim*250, 2500]))])+"\n"


with open("isotropy_measures_100dim.csv", "w") as f:
    f.write(sprow)
    f.write(circrow)
    f.write(ringrow)
    f.write(conerow)
    f.write(clustrow)
    f.write(clust_uerow)
    f.write(clust_srow)
    
#%%
#plot sphere comapre
f, axes = plt.subplots(2,2, figsize = (8,6))
f.tight_layout()
for i in range(2):
    for ax in axes[i]:
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.set_xlim([-1.05,1.05])
        ax.set_ylim([-1.05,1.05])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, which = 'both')

plt.subplots_adjust(hspace = 0.25)

axes[1,0].set_title('Shell', fontsize = 'xx-large')        
sns.scatterplot(x = circ[2].T[0], y = circ[2].T[1], ax = axes[1,0])
#axes[1,0].text(-0.44,-1.24,'Avg Cos=' +str(round(avg_cos(circ[2], num_sample = 500),4)), fontsize = 'xx-large')
#axes[1,0].text(-0.44,-1.46,'I(V)=' +str(round(Iw(circ[2]),4)), fontsize = 'xx-large')

axes[1,1].set_title('Nested Shell', fontsize = 'xx-large')        
sns.scatterplot(x = ring[2].T[0], y = ring[2].T[1], ax = axes[1,1])
#axes[1,1].text(-0.44,-1.24,'Avg Cos=' +str(round(avg_cos(ring[2], num_sample = 500),4)), fontsize = 'xx-large')
#axes[1,1].text(-0.44,-1.46,'I(V)=' +str(round(Iw(ring[2]),4)), fontsize = 'xx-large')

axes[0,1].set_title('Cone', fontsize = 'xx-large')        
sns.scatterplot(x = cone[2].T[0], y = cone[2].T[1], ax = axes[0,1])
#axes[0,1].text(-0.44,-1.24,'Avg Cos=' +str(round(avg_cos(cone[2], num_sample = 500),4)), fontsize = 'xx-large')
#axes[0,1].text(-0.44,-1.46,'I(V)=' +str(round(Iw(cone[2]),4)), fontsize = 'xx-large')

axes[0,0].set_title('Sphere', fontsize = 'xx-large')        
sns.scatterplot(x = sp[2].T[0], y = sp[2].T[1], ax = axes[0,0])
#axes[0,0].text(-0.44,-1.24,'Avg Cos=' +str(round(avg_cos(sp[2], num_sample = 500),4)), fontsize = 'xx-large')
#axes[0,0].text(-0.44,-1.46,'I(V)=' +str(round(Iw(sp[2]),4)), fontsize = 'xx-large')

plt.savefig("2d_sphere_clean.pdf", bbox_inches = 'tight')
#%%
#plot cluster compare
ymin = np.min([clust[2].T[1], clust_ue[2].T[1], clust_s[2].T[1]])
ymax = np.max([clust[2].T[1], clust_ue[2].T[1], clust_s[2].T[1]])

xmin = np.min([clust[2].T[0], clust_ue[2].T[0], clust_s[2].T[0]])
xmax = np.max([clust[2].T[0], clust_ue[2].T[0], clust_s[2].T[0]])

f, axes = plt.subplots(1,3, figsize = [8,6])
f.tight_layout()

for ax in axes:
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.grid(True, which = 'both')
    ax.set_ylim([ymin,ymax])
    ax.set_xlim([xmin,xmax])
        
plt.subplots_adjust(wspace = 0.1)    

axes[0].set_title('Symmetric', fontsize = 'xx-large')
sns.scatterplot(x = clust[2].T[0][:125], y = clust[2].T[1][:125], ax = axes[0], color= "C0")
sns.scatterplot(x = clust[2].T[0][125:250], y = clust[2].T[1][125:250], ax = axes[0], color = "C0")
sns.scatterplot(x = clust[2].T[0][250:375], y = clust[2].T[1][250:375], ax = axes[0], color = "C1")
sns.scatterplot(x = clust[2].T[0][375:500], y = clust[2].T[1][375:500], ax = axes[0], color = "C1")
#axes[0].text(-1.3,-2.10,'Avg Cos=' +str(round(avg_cos(clust[2], num_sample = 500),4)), fontsize = 'xx-large')
#axes[0].text(-1.3,-2.3,'I(V)=' +str(round(Iw(clust[2]),4)), fontsize = 'xx-large')

axes[1].set_title('Uneven', fontsize = 'xx-large')
sns.scatterplot(x = clust_ue[2].T[0][0:161],y = clust_ue[2].T[1][0:161], ax = axes[1], color = "C0")
sns.scatterplot(x = clust_ue[2].T[0][161:250],y = clust_ue[2].T[1][161:250], ax = axes[1], color = "C0")
sns.scatterplot(x = clust_ue[2].T[0][250:369],y = clust_ue[2].T[1][250:369], ax = axes[1], color = "C1")
sns.scatterplot(x = clust_ue[2].T[0][369:500],y = clust_ue[2].T[1][369:500], ax = axes[1], color = "C1")
#axes[1].text(-1.3,-2.10,'Avg Cos=' +str(round(avg_cos(clust_ue[2], num_sample = 500),4)), fontsize = 'xx-large')
#axes[1].text(-1.3,-2.3,'I(V)=' +str(round(Iw(clust_ue[2]),4)), fontsize = 'xx-large')
axes[1].text(-0.63946062-0.8, -0.96104952-0.1, "161", fontsize = "xx-large")
axes[1].text(0.63946062+0.3, 0.96104952+0.1, "89", fontsize = "xx-large")
axes[1].text(-0.07356295-0.8,  0.44986786-0.1, "119", fontsize = "xx-large")
axes[1].text(0.07356295+0.4,  -0.44986786-0.1, "131", fontsize = "xx-large")


axes[2].set_title('Shifted', fontsize = 'xx-large')
sns.scatterplot(x = clust_s[2].T[0][:125],y = clust_s[2].T[1][:125] ,ax = axes[2], color = "C0")
sns.scatterplot(x = clust_s[2].T[0][125:250],y = clust_s[2].T[1][125:250] ,ax = axes[2], color = "C0")
sns.scatterplot(x = clust_s[2].T[0][250:375],y = clust_s[2].T[1][250:375] ,ax = axes[2], color = "C1")
sns.scatterplot(x = clust_s[2].T[0][375:500],y = clust_s[2].T[1][375:500] ,ax = axes[2], color = "C1")
#axes[2].text(-1.3,-2.10,'Avg Cos=' +str(round(avg_cos(clust_s[2], num_sample = 500),4)), fontsize = 'xx-large')
#axes[2].text(-1.3,-2.3,'I(V)=' +str(round(Iw(clust_s[2]),4)), fontsize = 'xx-large')

plt.savefig("2d_clust_clean.pdf", bbox_inches = 'tight')
#%%   

models = [sp2,circ2,ring2,clust2,cone2,clust2_ue,clust2_s,
          sp10,circ10,ring10,clust10,cone10,clust10_ue,clust10_s,
          sp50,circ50,ring50,clust50,cone50,clust50_ue,clust50_s,
          sp100,circ100,ring100,clust100,cone100,clust100_ue,clust100_s]

model_names = ['sp2','circ2','ring2','clust2','cone2','clust2_ue','clust2_s',
          'sp10','circ10','ring10','clust10','cone10','clust10_ue','clust10_s',
          'sp50','circ50','ring50','clust50','cone50','clust50_ue','clust50_s',
          'sp100','circ100','ring100','clust100','cone100','clust100_ue','clust100_s']
    
for i in range(len(models)):
    print(model_names[i])
    print(knn_overlap(models[i]))
    
models = [sp50,circ50,ring50,sp100,circ100,ring100]
model_names = ['sp50','circ50','ring50','sp100','circ100','ring100']

for i in range(len(models)):
    print(model_names[i])
    print("pcr:", PC_ratio(models[i]))
    print("vasicek:", vasicek_entropy(models[i]))
    
#%%

q = np.random.normal(0,1,size = 500)
plt.figure()
for i in range(10):
    sns.kdeplot(clust_s[10].T[i], alpha = 0.5)
#plt.hist(clust2.T[1], alpha = 0.5, range = (min(q),max(q)), bins = 30)
plt.show()

#%%
sp = dict()
circ = dict()
ring = dict()
clust = dict()
cone = dict()
clust_ue = dict()
clust_s = dict()

for dim in [2,10,50,100]:
    np.random.seed(11)
    sp[dim] = sphere(dim)
    
    np.random.seed(11)
    circ[dim] = sphere(dim, fill = False)
    
    np.random.seed(11)
    ring[dim] = sphere(dim, fill = False, rings = 2)
    
    np.random.seed(11)
    clust[dim] = cluster(dim)
    
    np.random.seed(11)
    cone[dim] = sphere(dim, cone= True)
    
    np.random.seed(11)
    clust_ue[dim] = cluster(dim, shuffle = True)
    
    np.random.seed(11)
    clust_s[dim] = cluster(dim, shift = True)

#%%
measures = dict()
models = [#sp,circ,ring,
          #clust,
          cone,
          #clust_ue,clust_s
          ]
model_names = [#'sp','circ','ring',
               #'clust',
               'cone',
               #'clust_ue','clust_s'
               ]
dims = [2,10,50,100]

for i in range(len(models)):
    m = models[i]
    mn = model_names[i]
    measures[mn] = dict()
    for dim in dims:
        results = isotropy_measures(m[dim], cos_samples = np.min([dim*250, 2500]))
        measures[mn][dim] = results
        # kl = kl_divergence_discrete(m[dim])
        # measures[mn][dim]= kl
        
#%%
#plot AUC scores
models = [sp,
          #circ,
          #ring,
          cone,
          clust,
          #clust_ue,
          clust_s
          ]
model_names = ['Sphere',
               #'circ',
               #'ring',
               'Cone',
               'Symmetric Clusters',
               #'clust_ue',
               'Shifted Clusters'
               ]

dim = 100

f, axes = plt.subplots(2,2, figsize = [8,6], constrained_layout = True, sharex = True, sharey = True)
f.tight_layout()
plt.subplots_adjust(hspace = 0.25)    

for i in range(2):
    for ax in axes[i]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax.spines['bottom'].set_visible(False)
        #ax.spines['left'].set_visible(False)
# for ax in axes:
#     ax.spines['left'].set_position('zero')
#     ax.spines['bottom'].set_position('zero')
#     ax.spines['right'].set_color('none')
#     ax.spines['top'].set_color('none')
#     ax.grid(True, which = 'both')
#     ax.set_ylim([ymin,ymax])
#     ax.set_xlim([xmin,xmax])
        
#plt.subplots_adjust(wspace = 0.1)  

for i in range(4):
    if i in [0,1]:
        r = 0
    else:
        r = 1
        
    if i in [0,2]:
        c = 0
    else:
        c = 1
    
    m = models[i][dim]
    
    pc = PCA()
    pc.fit(m)
    num_pc = pc.n_components_
    eigensum = np.cumsum(pc.explained_variance_)
    eigensum /= eigensum[-1]
    ref = np.cumsum([eigensum[-1]/(num_pc)]*(num_pc))
    AUC_sum = eigensum-ref
    AUC = auc(range(num_pc),AUC_sum)
    
    total_poss = (eigensum[-1]*num_pc)/2
    measure = round(AUC/total_poss,4)
    
    axes[r,c].set_title(model_names[i], fontsize = 'xx-large')
    axes[r,c].plot(range(num_pc), eigensum, label = 'Cumulative Sum of Eigenvalues')
    axes[r,c].plot(range(num_pc), ref, linestyle = 'dashed')
    axes[r,c].text(num_pc/1.9, eigensum[-1]/13, 'EEE='+str(measure), fontsize = 'xx-large')


f.tight_layout(rect = (0.025,0.025,1,1), pad = 1.9)    
f.supxlabel('Principal Component', fontsize = 'xx-large')
f.supylabel('Normalized Eigenvalue Cumulative Sum', fontsize = 'xx-large')
    
#plt.show()

plt.savefig("auc_compare_clean.pdf" , bbox_inches = 'tight')

#%%
#visualize vasicek ratio

models = [sp,
          #circ,
          #ring,
          cone,
          clust,
          #clust_ue,
          clust_s
          ]
model_names = ['Sphere',
               #'circ',
               #'ring',
               'Cone',
               'Symmetric Clusters',
               #'clust_ue',
               'Shifted Clusters'
               ]

dim = 100

f, axes = plt.subplots(2,2, figsize = [8,6], constrained_layout = True, sharex = True, sharey = True)
f.tight_layout()
plt.subplots_adjust(hspace = 0.25)   

for i in range(2):
    for ax in axes[i]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim([0,1.8])
        ax.set_xlim([0,100])


for i in range(4):
    if i in [0,1]:
        row = 0
    else:
        row = 1
        
    if i in [0,2]:
        col = 0
    else:
        col = 1
    
    m = models[i][dim]
    N = len(m)
    d = len(m[0])
    pc = PCA()
    tx = pc.fit_transform(m)
    
    #fix total explained variance in data to match that of a standard normal distribution
    comp,sample, ev = normal_compare(d,N)
    scale = ev/sum(pc.explained_variance_)
    tx *= np.sqrt(scale)
     
    #compute theoretical max vasicek entropy for std normal comparison
    m = np.log(np.sqrt(2*np.pi*e))
    
    ent = differential_entropy(tx)
    r = np.exp(ent)/np.exp(m)

    ref = [1]*d
    rsum = -np.sort(-r)
    
    ss = round(np.sum((1-r)**2)/100,4)
    
    axes[row,col].set_title(model_names[i], fontsize = "xx-large")
    axes[row,col].plot(range(d), ref, linestyle = 'dashed', linewidth = 3, color = 'orange')
    axes[row,col].plot(range(d),r, label = 'Cumulative Sum of Eigenvalues', linewidth = 3)
    axes[row,col].text(15, 0.25, 'VRM='+str(ss), fontsize = "xx-large")


f.tight_layout(rect = (0.025,0.025,1,1), pad = 1.9)    
f.supxlabel('Principal Component', fontsize = "xx-large")
f.supylabel('Vasicek Entropy Ratio', fontsize = "xx-large")
    
#plt.show()
plt.savefig("vasicek_comp_clean.pdf", bbox_inches = 'tight')
#%%
#discrete kl divergence
models = [sp,
          #circ,
          #ring,
          cone,
          clust,
          #clust_ue,
          clust_s
          ]
model_names = ['Sphere',
               #'circ',
               #'ring',
               'Cone',
               'Symmetric Clusters',
               #'clust_ue',
               'Shifted Clusters'
               ]

dim = 100

f, axes = plt.subplots(2,2, figsize = [8,6], constrained_layout = True)
f.tight_layout()
plt.subplots_adjust(hspace = 0.25)   


np.random.seed(11)
for i in range(4):
    if i in [0,1]:
        row = 0
    else:
        row = 1
        
    if i in [0,2]:
        col = 0
    else:
        col = 1
        
    axes[row][col].spines['top'].set_visible(False)
    axes[row][col].spines['right'].set_visible(False)
    axes[row][col].set_xlim([-1,100]) 
    
    if i < 3:
        axes[row][col].set_ylim([0,2])
        axes[row,col].text(15, 1.25, 'DKLS ='+str(ss), fontsize = "xx-large")
    else:
        axes[row,col].text(15, 29, 'DKLS ='+str(ss), fontsize = "xx-large")
    
    m = models[i][dim]

    N = len(m)
    d = len(m[0])
    pc = PCA()
    tx = pc.fit_transform(m)
    
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

    ref = [0]*d
    rsum = -np.sort(-kl)
    
    ss = round(np.sum(kl**2)/100,4)
    
    axes[row,col].set_title(model_names[i], fontsize = 'xx-large')
    axes[row,col].plot(range(d), kl, linewidth = 3)
    

# f.add_subplot(111, frameon = False)
# plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
# plt.xlabel('Principal Component')
# plt.ylabel('Discrete KL-Divergence')
f.tight_layout(rect = (0.025,0.025,1,1), pad = 1.9)    
f.supxlabel('Principal Component', fontsize = 'xx-large')
f.supylabel('Discrete KL-Divergence', fontsize = 'xx-large')
    
#plt.show()

plt.savefig("discretekl_comp_clean.pdf", bbox_inches = 'tight')
    
#%%
#heatmap of dimensions
dmin = np.min(cone[10])
dmax = np.max(cone[10])

ht = np.array([np.histogram(cone[10].T[i], range = (dmin, dmax), bins = 25)[0] for i in range(10)])
labs = np.array([np.histogram(cone[10].T[i], range = (dmin, dmax), bins = 25)[1] for i in range(10)])
plt.figure()
sns.heatmap(ht.T, yticklabels = np.round(labs[0],4))
plt.show()

#%%
# ridgeline plots of dimensions


dim = 10

models = [sp,
          circ,
          ring,
          cone,
          clust,
          clust_ue,
          clust_s
          ]
model_names = ['Sphere',
               'Shell',
               'Nested Shell',
               'Cone',
               'Symmetric Clusters',
               'Uneven Clusters',
               'Shifted Clusters'
               ]


for i in range(7):
        
    df = pd.DataFrame(columns = ['dim', 'Dimension','value'])    
    d = pd.melt(pd.DataFrame(models[i][dim]), var_name = 'Dimension')
    d['dim'] = [2]*len(d)
    df = pd.concat([df,d])
    mn = model_names[i]

    #df = pd.melt(pd.DataFrame(m[dim]), var_name = 'Dimension')
        
    sns.set_theme(style = 'white', rc ={"axes.facecolor":(0,0,0,0)})
    g = sns.FacetGrid(df, row = 'Dimension',  aspect = 10, height =1)
    g.map_dataframe(sns.kdeplot, x = 'value', color = 'black')
    g.map_dataframe(sns.kdeplot, x = 'value', fill = True, alpha = 1)
    g.fig.subplots_adjust(hspace = -.2)
    g.set_titles("")
    g.set(yticks = [], ylabel = None, xlabel = None)
    g.despine(left = True)
    plt.suptitle(mn, fontsize = 50,y = 1.05)
    #plt.show()
    plt.savefig(f"{mn}_ridgeline_clean.pdf", bbox_inches = 'tight')
    

#%%
np.random.seed(11)
cone2 = sphere(2,cone = True)

np.random.seed(11)
cone2_2 = sphere(2, cone = True)

#%%
#knn overlap viz

models = [sp,
          #circ,
          #ring,
          cone,
          clust,
          clust_ue,
          #clust_s
          ]
model_names = ['Sphere',
               #'circ',
               #'ring',
               'Cone',
               'Symmetric Clusters',
               'Uneven Clusters',
               #'Shifted Cluster'
               ]

dim = 100

f, axes = plt.subplots(2,2, figsize = [8,6], constrained_layout = True)
f.tight_layout()
plt.subplots_adjust(hspace = 0)   
np.random.seed(11)
for i in range(4):
    if i in [0,1]:
        row = 0
    else:
        row = 1
        
    if i in [0,2]:
        col = 0
    else:
        col = 1
        
    embeddings = models[i][dim]
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
    
    
    
    rho_1_bin = np.histogram(rho_1, bins = 30, range=(rho_min, rho_max))
    rho_k_bin = np.histogram(rho_k, bins = 30, range = (rho_min, rho_max))
    rho_difference = rho_k_bin[0] - rho_1_bin[0]
    rho_ovl =  rho_k_bin[0]-[i if i >0 else 0 for i in rho_difference]
    
    max_rho_bin = max(max(rho_1_bin[0]), max(rho_k_bin[0]))
    
    ovl_perc = round(np.sum(rho_ovl)/np.sum(rho_k_bin[0]),4)
    axes[row,col].set_xlim([rho_min-0.1*rho_min, rho_max+0.1*rho_max])
    axes[row,col].set_ylim([0, max_rho_bin + 0.25*max_rho_bin])
    axes[row][col].spines['top'].set_visible(False)
    axes[row][col].spines['right'].set_visible(False)
    axes[row,col].set_title(model_names[i], fontsize = 'xx-large')
    axes[row,col].hist(rho_1,bins = 30,range=(rho_min, rho_max), label = 'First Nearest Neighbors', histtype = 'step', linewidth = 3)
    axes[row,col].hist(rho_k,bins = 30,range=(rho_min, rho_max), label = f'{k}th Nearest Neighbors', histtype = 'step', linewidth = 3)
    axes[row,col].text(rho_min-0.07*rho_min,max_rho_bin+0.15*max_rho_bin,'KNN Ovl ='+str(ovl_perc), ha='left', va = 'top', fontsize = 'xx-large')
    # else:
    #     axes[row,col].text(rho_min,max_rho_bin,'KNN Ovl = '+str(ovl_perc), ha='left', va = 'top', fontsize = 'xx-large')


f.tight_layout(rect = (0.025,0.025,1,1), pad =1.9, w_pad = 0.2, h_pad = 1)    
f.supxlabel('KNN Euclidean Distance', fontsize = 'xx-large')
f.savefig("KNNovl_clean.pdf", bbox_inches = "tight")
    
#%%

#normal comparison
models = [normal_compare(2,500)[1],
          normal_compare(10,2500)[1],
          normal_compare(50, 12500)[1],
          normal_compare(100, 25000)[1]]
          
measures = []

for i in range(len(models)):
    m = models[i]
    results = isotropy_measures(m, cos_samples = np.min([len(m[0])*250, 2500]))
    measures.append( results)        

print(measures[3])
