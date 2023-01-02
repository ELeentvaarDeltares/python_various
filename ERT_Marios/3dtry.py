# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 12:54:32 2023

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

yt = glob.glob(r"C:\Users\leentvaa\python_various\ERT_Marios\test3d\*.data")

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
    make_pyg_3d('C:\\Users\\leentvaa\\python_various\\ERT_Marios\\pyg\\lin1_%04d.pyg'%i, new_elec, tt)
    
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
    world = mt.createCube(size=[7.75, 1.5, -2], start=[0,0,-2.0], end=[7.75,1.5, 0], boundaryMarker=1)
    pg.show(world)
    pos_xy=np.array(pos)
    pos_xy=np.c_[pos_xy[:,0],pos_xy[:,1],pos_xy[:,2]] # array with positions of only x and y coordinates electrodes

    geom = world 
    for po in pos_xy:
        geom.createNode(po, marker=-99)# Markers on the position of the electrodes
        geom.createNode([po[0],po[1]-0.005,po[2]], marker=-99)  # Markers on the position 0.005 below electrodes for better resolution mesh
    geom.createNode([3, 0.75,-1.5], marker=-999)  #  reference electrode position inside the PLC, with a marker -999, somewhere away from the electrodes
    geom.createNode([7.75, 1.5, -2], marker=-1000) # calibration node with marker -1000 where the potential is fixed , somewhere on the boundary and far from the electrodes
    geom.createNode([0.1, 0.1, 0-1e-3]) #reference marker
        #Make these last ones automatic
    mesh = mt.createMesh(geom, quality=1.2, area=0.5) 
    #Area = Maximum element size (global) 
    #Quality = 2D triangle quality sets a minimum angle constraint
    
    # Show mesh and geometry for check 
    pg.show(mesh, showMesh=True)
    pg.show(geom)

    return mesh

# TODO, read files in certain folder once and automatically pick one for mesh making and directly invert the rest afterwards. 
data = ert.load("C:\\Users\\leentvaa\\python_various\\ERT_Marios\\pyg\\lin1_0000.pyg") # Load previously made pyg files. This one is for making the mesh.

# pg.viewer.mpl.showDataContainerAsMatrix(data, "a", "m", "rhoa",cMin=100,cMax=5000,cmap='rainbow')
data["k"] = ert.geometricFactors(data)  # calculates configuration factors for ert data, this one has someting to do with topography
data["err"] = ert.estimateError(data) # calculates configuration factors for ert data

mesh = makeMesh(data.sensors()) # Make the mesh with electrode positions. 



import glob

yt=glob.glob('C:\\Users\\leentvaa\\python_various\\ERT_Marios\\pyg\\*.pyg')


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
  