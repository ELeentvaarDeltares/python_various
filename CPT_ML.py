# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 09:34:05 2022

@author: leentvaa
"""
# Import packages

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import BisectingKMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch

from pygef import Cpt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mc
import cmapy
import random

from sklearn.linear_model import Ridge

from sklearn.decomposition import PCA

#%%

# Lithology value calc
def calc_ic(qc, rf) -> np.ndarray:
    return np.sqrt((3.47 - np.log10(qc / 0.1)) ** 2 + (np.log10(rf) + 1.22) ** 2)

# Read CPT data
def CPT_for_ML(path_to_files):
    with open(path_to_files, encoding="utf-8", errors="ignore") as f:
        s = f.read()
    
    gef = Cpt(content=dict(string=s, file_type="gef"))
    
    
    # Get info from the CPT file
    depth = gef.df["depth"]
    data_qc = np.expand_dims(gef.df["qc"], axis=1)
    data_friction = np.expand_dims(gef.df["fs"], axis=1)
    data_rf = (data_friction/data_qc)*100
    data_lith = calc_ic(data_qc, data_rf)
    
    # Prepare cpt_data for ML
    data_cpt = np.concatenate((data_friction, data_qc, data_lith, data_rf), axis=1)
    
    
    # Plot CPT data
    plt.plot(data_cpt[::-1, 0], np.array(depth), label='friction')
    plt.plot(data_cpt[::-1, 1], np.array(depth), label='qc')
    plt.plot(data_cpt[::-1, 2], np.array(depth), label='lith')
    plt.plot(data_cpt[::-1, 3], np.array(depth), label='rf')
    plt.legend()
    plt.grid()
    plt.title('CPT Data')
    plt.show()
    
    
    return data_cpt, depth

def PlotMLresults(data, name):
    # Number of clusters in labels, ignoring noise if present.
    labels = data.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    # Plot DBSCAN
    colors = {}
    unique_users = list(set(labels)) 
    step_size = 1 / len(unique_users)
    for i, user in enumerate(unique_users):
        colors[user] = i*step_size# cmapy.color('viridis', random.randrange(0,1), rgb_order=True)

    labelsdf = pd.DataFrame(labels[::-1], columns=['colorcode'])

    plt.plot(data_cpt[::-1, 0], np.array(depth).T,
             label='friction', color='black', zorder=1)
    plt.plot(data_cpt[::-1, 1], np.array(depth).T,
             label='qc', color='black', zorder=1)
    plt.plot(data_cpt[::-1, 2], np.array(depth).T,
             label='lith', color='black', zorder=1)
   # plt.plot(data_cpt[::-1, 3], np.array(depth).T,
    #         label='rf', color='black', zorder=1)
    #plt.scatter(data_cpt[::-1, 0], np.array(depth).T,
    #            c=labelsdf['colorcode'].map(colors), s=7, zorder=2)
    #plt.scatter(data_cpt[::-1, 1], np.array(depth).T,
    #            c=labelsdf['colorcode'].map(colors), s=7, zorder=2)
    plt.scatter(-10*np.ones(len(data_cpt[::-1, 1])),np.array(depth).T,c=labelsdf['colorcode'].map(colors),marker='_', s=300)
    #plt.legend()
    plt.grid()
    plt.title(str(name))
    plt.show()
    return

#%%
# Data
data_cpt, depth = CPT_for_ML("N:\\Projects\\11208000\\11208020\\B. Measurements and calculations\\053_unsupervised_facies_classification\\00_data\\02_cpt_data_reports\\CPT4a.GEF")

print(np.where(data_cpt >1000))
data_cpt[20,2] = 30
data_cpt[398,2] = 30
data_cpt[399,2] = 30
# %%

# Clustering methods:
    
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=5).fit(data_cpt)
db_results = PlotMLresults(db, 'DBSCAN')

# %%
# Compute k-means
km = KMeans(n_clusters=3, random_state=0).fit(data_cpt)
km_results = PlotMLresults(km, 'Kmeans')

# %%
# Compute affinity propagation
ap = AffinityPropagation(preference=-50, random_state=0).fit(data_cpt)
ap_results = PlotMLresults(ap, 'Affinity Propagation')

# %% 
# Compute Bisecting KMeans
bkm =  BisectingKMeans(n_clusters=3, random_state=0).fit(data_cpt)
bkm_results = PlotMLresults(bkm,'Bisecting KMeans')

#%%
#BIRCH
brc = Birch(n_clusters=5).fit(data_cpt)
brc_results = PlotMLresults(brc, 'BIRCH')

#%%
# Unsupervised dimensionality reduction
aa = AgglomerativeClustering(n_clusters=8).fit(data_cpt)
aa_results = PlotMLresults(aa, 'aa')

#%%

#RBEAST (Idea Marios)
import Rbeast as rb

ndvi, year, datestr = rb.load_example('ndvi')
print(np.shape(data_cpt))
metadata      = rb.args()         # create an empty object to stuff the attributes: "metadata  = lambda: None" also works
metadata.season       = 'none'    # the period is 1.0 year, so freq= 1.0 /(1/12) = 12 data points per period
metadata.whichDimIsTime = 1  
metadata.isRegularOrdered = True
metadata.startTime = 0
metadata.deltaTime = 0.01

o0 = rb.beast123(data_cpt,metadata)
rb.plot(o0)


#%%

#Ruptures (idea Marios)
import ruptures as rpt

algo = rpt.Pelt(model='rbf').fit(data_cpt)
result = algo.predict(pen=10)
rpt.display(data_cpt, result)
plt.show()