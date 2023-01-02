# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:18:26 2022

@author: obandohe
"""

import os
import glob
import numpy as np
from obspy import read
from segypy import wiggle
import matplotlib
import matplotlib.pylab as plt
from scipy.io import savemat
matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14)
import matplotlib.pyplot as plt

path0 = r'C:\Users\leentvaa\PythonScripts\Refraction_Seismic\to_Eline\to_Eline'
os.chdir(path0)

import masw_processing as mpross 


# Index to select traces at 0.5 m spacing
index = np.arange(0,60)          
# Index to select traces at 1.0 m spacing.

shots =  np.arange(0,17)
coord_x = np.arange(-0.5,8.0,0.5)

shots_idx = {0:[1000,1002],
             1:[1003,1005],
             2:[1006,1008],
             3:[1009,1011],
             4:[1012,1014],
            5:[1015,1017],
            6:[1018,1020],
            7:[1021,1023],
            8:[1024,1026],
            9:[1027,1029],
            10:[1030,1032],
            11:[1036,1038],
            12:[1040,1042],
            13:[1043,1045], 
            14:[1046,1048],
            15:[1049,1051],
            16:[1052,1054]}
            


shots_idx  = shots_idx


rec_names = {}
for ip in range(len(coord_x)): 
    rec = shots_idx[shots[ip]]
    rec_names[ip] = [str(idx0)+".dat" for idx0 in range(rec[0],rec[1]+1)]  
    
  
  #%%  
nn = 0

for idx in range(17):
    
    #idx = 3
    seg2data =  rec_names[shots[idx]] 
    path = r'D:\LAB_EXPERIMENT_UG_21112022'
    os.chdir(path)
    GEODEFiles = seg2data
    
    
    data_all = np.zeros((500,59,len(GEODEFiles)))
    
    for jj, sg2 in enumerate(GEODEFiles):
        # Search for all tdms files contained in the directory.    
        st=read(sg2)   # Obspy function to open SEG2 data format
        
        st.filter("lowpass",freq=500,zerophase=True)
        
        size=np.shape(st)
        data=np.zeros((size[1], size[0]))
        for i in range(size[0]):
            tr=np.reshape(st[i],(1,size[1]))
            data[:, i] = tr
        fs = st[0].stats.sampling_rate  
        
        data_all[:,:,jj] = data[350:850,:59]
    
    
    data_mean = np.mean(data_all,axis=2)
    data_norm =  data_mean/np.max(np.abs(data_mean),axis=0)
    wiggle(data_norm,gain=1.5)
    #plt.ylim(1200,350)
    
    mdic = {'data': data_mean}
    path = r'D:\DELTARES_PROJECTS\2022\01_URBAN_GEOPHYSICS\LAB_EXPERIMENT\00_REFRACTION'
    os.chdir(path)   
    savemat('shot_'+str(nn).zfill(3) +'.mat', mdic)
    
    nn+=1

