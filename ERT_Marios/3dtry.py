# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 13:38:11 2022

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



# Find electrode location in array
def sel_elec(data,first,last):
   
    i1 = np.where( (data[:,0]>=first) & (data[:,0]<=last)&
                (data[:,1]>=first) & (data[:,1]<=last)&
                (data[:,2]>=first) & (data[:,2]<=last)&
                (data[:,3]>=first) & (data[:,3]<=last))[0]
                
    return i1


# Import .data files -- these data files should be the same size!!
# TODO: Make function that automatically descard data files with zeros or those that are terminated early
yt = glob.glob(r"P:\11208018-016-urban-geophysics\4. Data\ERT\T3-Warm_left\*.data")

# Loop over files
# TODO: make if statement for when there is only one file. 
for i in range(0,len(yt)): 
    # Read mpt data module
    data, elec = read_mpt_data(yt[i])
    
    if i==0: # For first file
        out=data[:,5]
    else: # Add other files to the out file
        out=np.c_[out,data[:,5]]



# Fix two missing electrodes
new_elec=np.r_[
    elec[:31,:],
    np.c_[32,7.75,0,0],
    elec[31:62,:],
    np.c_[64,7.75,0.5,0],
    elec[62:,:]]

# Remove counter
new_elec = new_elec[:,1:]


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


abmn = np.c_[data[:,:4], out] 


# Find data locations in the data file that are related to different electrode
# lines. This can be used to split data into different lines
d3 = sel_elec(data,65,128)




# For every data line
# in the make pyg file, there is new elec, these are the locations of the electrodes. in tt are the abmn and resistivity files. 

for i in range(0,out.shape[1]):
    iiii = np.where(out[d3,i]>-np.inf)[0]  #Find data that is not infinity
    tt = np.c_[data[d3[iiii],:4],out[d3[iiii],i]] # merge data with 

    tt[:,0:4] = tt[:,0:4]-79
   
    make_pyg_3d('P:\\11208018-016-urban-geophysics\\4. Data\\ERT\\T3-Warm_left\\pyg3d\\lin1_%04d.pyg'%i, new_elec[:64,:], tt)
    


import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
import numpy as np
import matplotlib.pyplot as plt

#%%
# Make mesh function
def makeMesh(pos, invbound=2, bound=10):
    """Generate mesh around electrodes."""
    
    # This is the part were you create a 2D domain. 
    # This is the example for the urban geophysics project with a known domain size
    apos = np.array(pos)
    world = mt.createCube(size = [7.75,1.5,-2], pos=[7.75/2, 0.75, -1], boundaryMarker=1)
    #world = mt.createCube(start=[0,0,-2.0], end=[7.75,1.5, 0], boundaryMarker=1)
    #world = mt.createWorld(start=[0,0,-2.0], end=[7.75,1.5, 0], boundaryMarker=1)
    #world = mt.createWorld(start=[min(apos[:,0]),min(apos[:,1]),min(apos[:,2])], end = [max(apos[:,0]),max(apos[:,1]),max(apos[:,2])])
    #world = mt.createCube(start=[min(apos[:,0]),min(apos[:,1]),min(apos[:,2])], end = [max(apos[:,0]),max(apos[:,1]),max(apos[:,2])])
    world.exportVTK("world.vtk")  # vtk
    
    pos_xy=np.array(pos)
    pos_xy=np.c_[pos_xy[:,0],pos_xy[:,1],pos_xy[:,2]] # array with positions of only x and y coordinates electrodes
    
    geom = world 
    for po in pos_xy:
        geom.createNode(po, marker=-99)# Markers on the position of the electrodes
        geom.createNode([po[0],po[1],po[2]-0.005], marker=-99)  # Markers on the position 0.005 below electrodes for better resolution mesh
    geom.createNode([3, 0.75,-1.5], marker=-999)  #  reference electrode position inside the PLC, with a marker -999, somewhere away from the electrodes
    geom.createNode([7.75,1.5,-2], marker=-1000) # calibration node with marker -1000 where the potential is fixed , somewhere on the boundary and far from the electrodes
    geom.createNode([0.5, 0.5, -0.5 - 1e-3]) #reference marker  
    #Make these last ones automatic
    
    mesh = mt.createMesh(geom, quality = 1.2, area = 0.005) 
    #Area = Maximum element size (global) 
    #Quality = 2D triangle quality sets a minimum angle constraint
    return mesh

# TODO, read files in certain folder once and automatically pick one for mesh making and directly invert the rest afterwards. 
data = ert.load("P:\\11208018-016-urban-geophysics\\4. Data\\ERT\\T3-Warm_left\\pyg3d\\lin1_0000.pyg") # Load previously made pyg files. This one is for making the mesh.

# pg.viewer.mpl.showDataContainerAsMatrix(data, "a", "m", "rhoa",cMin=100,cMax=5000,cmap='rainbow')

data["k"] = ert.geometricFactors(data)  # calculates configuration factors for ert data, this one has someting to do with topography
data["err"] = ert.estimateError(data) # calculates configuration factors for ert data

mesh = makeMesh(data.sensors()) # Make the mesh with electrode positions. 

#%%

import glob

yt=glob.glob('P:\\11208018-016-urban-geophysics\\4. Data\\ERT\\T3-Warm_left\\pyg3d\\lin1_00*.pyg')

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
