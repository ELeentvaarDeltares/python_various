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

from pygef import Cpt, Bore
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mc
import cmapy
import random

from matplotlib.patches import Rectangle
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import sys

sys.path.insert(1, 'C:\\Users\\leentvaa\\pysst')
from pysst.ml.preprocessing import calc_machine_learning_params
from pathlib import Path
from pysst import read_gef_cpts

#%%

# Lithology value calc
def calc_ic(qc, rf) -> np.ndarray:
    return np.sqrt((3.47 - np.log10(qc / 0.1)) ** 2 + (np.log10(rf) + 1.22) ** 2)


def PlotMLresults(data, name, data_cpt,depth):
    # Number of clusters in labels, ignoring noise if present.
    labels = data.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    #print("Estimated number of clusters: %d" % n_clusters_)
    #print("Estimated number of noise points: %d" % n_noise_)

    # Plot DBSCAN
    colors = {}
    unique_users = list(set(labels)) 
    step_size = 1 / len(unique_users)
    for i, user in enumerate(unique_users):
        colors[user] = i*step_size# cmapy.color('viridis', random.randrange(0,1), rgb_order=True)

    labelsdf = pd.DataFrame(labels[::-1], columns=['colorcode'])
    
    plt.plot(data_cpt['fs'], np.array(depth),
             label='friction', color='white', zorder=1)
    plt.plot(data_cpt['qc'], np.array(depth),
              label='qc', color='white', zorder=1)
    plt.plot(data_cpt['ic'], np.array(depth),
              label='ic', color='white', zorder=1)
    plt.plot(data_cpt['dqc'], np.array(depth),
            label='dqc', color='white', zorder=1)
    plt.plot(data_cpt['dqc2'], np.array(depth),
            label='dqc2', color='white', zorder=1)
    plt.plot(data_cpt['dfs'], np.array(depth),
            label='dfs', color='white', zorder=1)
    plt.plot(data_cpt['dfs2'], np.array(depth),
            label='dfs2', color='white', zorder=1)
    plt.plot(data_cpt['volatility'], np.array(depth),
            label='volatility', color='white', zorder=1)
   
    plt.scatter(np.zeros(depth.shape[0]),np.array(depth),c=labelsdf['colorcode'].map(colors),marker='_', s=4*(abs(min(data_cpt.min('numeric_only'==True)))+abs(max(data_cpt.max('numeric_only'==True)))),zorder=0)
    
    #plt.legend()
    plt.grid()
    plt.title(str(name))
    plt.show()
    return

def multiplot(db,km,ap,bkm,brc,aa,rbeast,rupt,depth,data_cpt):
    fig, axs = plt.subplots(nrows=1, ncols=6)

    allresults = [db,km,ap,bkm,brc,aa,0,1]
    name_temp = ['db','km', 'ap', 'bkm','brc', 'aa']
    
    counter=1
  
    for data,ax in zip(allresults,axs.ravel()):
        
        if data == 0:
            print('fix later')
            
        else:
            counter+=1
            labels = data.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)
    
            colors = {}
            unique_users = list(set(labels)) 
            step_size = 1 / len(unique_users)
            for i, user in enumerate(unique_users):
                colors[user] = i*step_size
    
            labelsdf = pd.DataFrame(labels[::-1], columns=['colorcode'])
            
            ax.scatter(np.zeros(depth.shape[0]),np.array(depth),c=labelsdf['colorcode'].map(colors),marker='_', s=100 ,zorder=0)
            ax.set_title((name_temp[counter-2]))
        #ax.scatter(np.zeros(depth.shape[0]),np.array(depth),c=labelsdf['colorcode'].map(colors),marker='_', s=100,zorder=0)
        if counter == 2:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(True)
        else:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
    return
    
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 16:16:10 2022

@author: knaake
"""

def create_patches(stratbounds, cpt_id, ax_obj, cmap='tab20'):
    
    sel = stratbounds[['CODE', cpt_id]].dropna(subset=[cpt_id])
    
    depth = sel[cpt_id]
    labels = sel['CODE']
    
    thickness = np.diff(depth)
    
    left, right = ax_obj.get_xlim()
    width = right - left
    
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(len(thickness)))
    
    for d, t, l, c in zip(depth[:-1], thickness, labels[:-1], colors):
        yield Rectangle((left, d), width, t, label=l, color=c, alpha=0.5)


def plot_strat_patches(ax_obj, stratbounds, cpt_id, cmap='tab20'):
    
    for p in create_patches(stratbounds, cpt_id, ax_obj, cmap):
        ax_obj.add_patch(p)
    return

#%%

# Load data 

# workdir = Path(r'c:\Users\knaake\OneDrive - Stichting Deltares\Documents\borehole_ml\testdata')
workdir = Path(r'n:\Projects\11208000\11208020\B. Measurements and calculations\053_unsupervised_facies_classification\00_data\02_cpt_data_reports')

stratbounds = pd.read_csv(workdir/'boundaries.csv', sep=';')
stratbounds.columns = stratbounds.columns.str.upper()


cpts = read_gef_cpts(workdir, use_standard_pygef=False)
cpts.data['nr'] = cpts.data['nr'].str.replace('YANGTZEHAVEN ', '')
cpts.data['nr'] = cpts.data['nr'].str.replace(' ', '')

cpts.data['fs'] = np.abs(cpts.data['fs'])
cpts.data.loc[cpts.data['fs']==0, 'fs'] = 1e-4 # resistance can't be smaller than 0
cpts.data['friction_number'] = (cpts.data['fs']/cpts.data['qc']) * 100

cpts.add_ic()
data_cpt = cpts.data.copy()
data_cpt = data_cpt.groupby('nr').apply(calc_machine_learning_params)

cpt = data_cpt[data_cpt['nr']=='AM436']   

ml_cols = ['qc', 'fs', 'ic', 'dqc', 'dqc2', 'dfs', 'dfs2', 'volatility']
depth = cpt['corrected_depth']

 
# %%

# Clustering methods:

# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=5).fit(cpt[ml_cols].values)
db_results = PlotMLresults(db, 'DBSCAN', cpt[ml_cols],depth)

# %%
# Compute k-means
km = KMeans(n_clusters=3, random_state=0).fit(cpt[ml_cols].values)
km_results = PlotMLresults(km, 'Kmeans', cpt[ml_cols],depth)

# %%
# Compute affinity propagation
ap = AffinityPropagation(preference=-50, random_state=0).fit(cpt[ml_cols].values)
ap_results = PlotMLresults(ap, 'Affinity Propagation', cpt[ml_cols],depth)

# %% 
# Compute Bisecting KMeans
bkm =  BisectingKMeans(n_clusters=3, random_state=0).fit(cpt[ml_cols].values)
bkm_results = PlotMLresults(bkm,'Bisecting KMeans', cpt[ml_cols],depth)
#%%
#BIRCH
brc = Birch(n_clusters=5).fit(cpt[ml_cols].values)
brc_results = PlotMLresults(brc, 'BIRCH', cpt[ml_cols],depth)

#%%
# Unsupervised dimensionality reduction
aa = AgglomerativeClustering(n_clusters=8).fit(cpt[ml_cols].values)
aa_results = PlotMLresults(aa, 'aa', cpt[ml_cols],depth)

#%%

#RBEAST (Idea Marios)
import Rbeast as rb

ndvi, year, datestr = rb.load_example('ndvi')

metadata      = rb.args()         # create an empty object to stuff the attributes: "metadata  = lambda: None" also works
metadata.season       = 'none'    # the period is 1.0 year, so freq= 1.0 /(1/12) = 12 data points per period
metadata.whichDimIsTime = 1  
metadata.isRegularOrdered = True
metadata.startTime = 0
metadata.deltaTime = 0.01

o0 = rb.beast123((cpt[ml_cols].values),metadata)
rb.plot(o0)


#%%

#Ruptures (Idea Marios)
import ruptures as rpt

algo = rpt.Pelt(model='rbf').fit(cpt[ml_cols].values)
result = algo.predict(pen=10)
rpt.display((cpt[ml_cols].values), result)
plt.show()

#%%
# multiplot sklearn methods
rbeast = 0
rupt = 0
res = multiplot(db,km,ap,bkm,brc,aa,o0,rupt,depth,data_cpt)
