# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 11:41:47 2023

@author: leentvaa
"""



# import standard libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import glob
import sys

# import pygimli packages 
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert

# import local packages 
sys.path.insert(0, r'C:/Users/leentvaa/python_various/ERT_Marios/')
from read_mpt_data import read_mpt_data
from make_pyg import make_pyg
from geofac2 import geofac2
from make_pyg_3d import make_pyg_3d

def sel_elec(data,first,last):
   
    i1 = np.where( (data[:,0]>=first) & (data[:,0]<=last)&
                (data[:,1]>=first) & (data[:,1]<=last)&
                (data[:,2]>=first) & (data[:,2]<=last)&
                (data[:,3]>=first) & (data[:,3]<=last))[0]
                
    return i1

#%% 
# FIRST TEST FOIL
# Import .data files -- these data files should be the same size!!
# TODO: Make function that automatically descard data files with zeros or those that are terminated early
yt = glob.glob(r'E:/Foil2022/Foil detection 70 cm/*.data')
print(yt)
# Loop over files
# TODO: make if statement for when there is only one file. 
for i in range(0,len(yt)): 
    # Read mpt data module
    data, elec = read_mpt_data(yt[i])

    if i==0: # For first file
        out=data[:,5]
    else: # Add other files to the out file
        out=np.c_[out,data[:,5]]


# Remove counter
new_elec = elec[:,1:]
# Custom array for foil detection

new_elec = np.array([[2., 0., -0.70],
            [2.1, 0., -0.70],
            [2.2, 0., -0.70],
            [2.3, 0., -0.70],
            [2.4, 0., -0.70],
            [2.5, 0., -0.70],
            [2.7, 0., -0.70],
            [0. ,0. ,0.],
            [0. ,0. ,0.],
            [0. ,0. ,0.],
            [3.7, 0., -0.5],
            [4.7, 0., -0.5],
            [5.7, 0., -0.5],
            [6.7, 0., -0.5]])


# coor = coordinates, first line is x , second line is y, third line = z, fourth line is ??
coor = np.c_[new_elec[data[:,0].astype('int32')-1],
           new_elec[data[:,1].astype('int32')-1],
           new_elec[data[:,2].astype('int32')-1],
           new_elec[data[:,3].astype('int32')-1]]

# Calculate geometrical factor 
# TODO WHAT IS GF AND GF2 DIFFERENCE
gf, gf2 = geofac2(coor)

# DONTKNOWYET
for i in range(0,out.shape[1]):
    out[:,i] = out[:,i]*gf
    
# A, B, M, N, lots of resistivities ?????
abmn = np.c_[data[:,:4], out] 


for i in range(0,out.shape[1]):
    iiii = np.where(out[:,i]>-np.inf)[0]  #Find data that is not infinity
    tt = np.c_[data[iiii,:4],out[iiii,i]] # merge data with 
    make_pyg_3d('E:\Foil2022\Foil detection 70 cm\lin1_%04d.pyg'%i, new_elec, tt)

#%%
# SECOND TEST FOIL



#%%


import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
import numpy as np
import matplotlib.pyplot as plt


# Make mesh function
def makeMesh(pos, invbound=2, bound=10):
    """Generate mesh around electrodes."""
    
    # This is the part were you create a 2D domain. 
    # This is the example for the urban geophysics project with a known domain size
    world = mt.createWorld(start=[0, -2.0],
                           end=[7.75, 0])
    pos_xy=np.array(pos)
    pos_xy=np.c_[pos_xy[:,0],pos_xy[:,1],pos_xy[:,2]] # array with positions of only x and y coordinates electrodes

    geom = world
    for po in pos_xy:
        geom.createNode(po, marker=-99)
        geom.createNode([po[0],po[1]-0.005], marker=-99)

        # geom.createNode(po - pg.Pos(0, 0.005))  # refinement

    # geom.exportPLC("mesh.poly")  # tetgen
    # geom.exportVTK("geom.vtk")  # vtk
    mesh = mt.createMesh(geom, quality=1.2,area=0.005)

    return mesh

# TODO, read files in certain folder once and automatically pick one for mesh making and directly invert the rest afterwards. 
data = ert.load("E:\Foil2022\Foil detection 70 cm\lin1_0000.pyg") # Load previously made pyg files. This one is for making the mesh.

# pg.viewer.mpl.showDataContainerAsMatrix(data, "a", "m", "rhoa",cMin=100,cMax=5000,cmap='rainbow')
data["k"] = ert.geometricFactors(data)  # calculates configuration factors for ert data, this one has someting to do with topography
data["err"] = ert.estimateError(data) # calculates configuration factors for ert data

mesh = makeMesh(data.sensors()) # Make the mesh with electrode positions. 



import glob
from pygimli import Inversion
yt=glob.glob('E:\Foil2022\Foil detection 70 cm\*.pyg')


for i in range(0,len(yt)):
    data = ert.load(yt[i])
    data["k"] = ert.geometricFactors(data)
    data["err"] = ert.estimateError(data)
    mgr = ert.ERTManager(data, verbose=False)
    mgr.invert(mesh=mesh,maxIter=5)
    mgr.saveResult(folder=yt[i][:-4])
    np.savetxt('pgr.txt',[i])
    plt.close()
    # mgr.showFit()
