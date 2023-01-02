# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 10:41:21 2023

@author: leentvaa
"""

import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
import numpy as np
import matplotlib.pyplot as plt

# Make mesh function
def makeMesh(pos, invbound=2, bound=10):
    """Generate mesh around electrodes."""
    xmin, xmax = min(pg.x(pos)), max(pg.x(pos)) # Read x and y positions from data file
    ymin, ymax = min(pg.z(pos)), max(pg.z(pos))

    xmid = (xmin+xmax)/2    # find the middle of the domain
    ymid = (ymin+ymax)/2    
    
    # This is the part were you create a 2D domain. 
    # This is the example for the urban geophysics project with a known domain size
    world = mt.createWorld(start=[0, -2.0],
                           end=[7.75, 0])
    # This is an example for the salt water intrusion project with less known boundaries.
    # not sure how they determined the boundaries, and why they use xmid and not xmin and xmax
    #world = mt.createWorld(start=[xmid-bound, -bound/2],
    #                       end=[xmid+bound, 0])
    # world.translate([xmid, ymid, 0])  # some bug in createWorld! # E: Not sure what this does.

    # DONT KNOW YET but noting is done with this part. ---------------
    maxdep = min(pg.z(pos)) - invbound
    # DONT KNOW WHAT INV BOUND IS
    sx = xmax - xmin + invbound * 2
    sy = ymax - ymin + invbound * 2

    # box=mt.createRectangle(start=[xmin-1, ymin], end=[xmax+1, 0],
    #                       marker=2,area=0.1)
    
    verts=[[xmax,0],[xmin,0]]
    temp=np.array(pos)
    for i in range(0,len(temp)):
        verts.append([temp[i,0],temp[i,2]-0.1])
    box=mt.createPolygon(verts,isClosed=True,marker=2,area=0.005)
    # box=mt.createRectangle(start=[xmin-1, ymin], end=[xmax+1, 0],
    #                       marker=2,area=0.1)
    # ---------------------------------------------------------------------
    
    pos_xy=np.array(pos)
    pos_xy=np.c_[pos_xy[:,0],pos_xy[:,1]] # array with positions of only x and y coordinates electrodes
    geom = world 
    for po in pos_xy:
        
        geom.createNode(po, marker=-99)# Markers on the position of the electrodes
        geom.createNode([po[0],po[1]-0.005], marker=-99)  # Markers on the position 0.005 below electrodes WHYYYY
        #

        # geom.createNode(po - pg.Pos(0, 0.005))  # refinement

    # geom.exportPLC("mesh.poly")  # tetgen
    # geom.exportVTK("geom.vtk")  # vtk
    
    mesh = mt.createMesh(geom, quality=1.2, area=0.005) 
    #Area = Maximum element size (global) 
    #Quality = 2D triangle quality sets a minimum angle constraint
    
    # Show mesh and geometry for check 
    pg.show(mesh)
    pg.show(geom)
    return mesh

# TODO, read files in certain folder once and automatically pick one for mesh making and directly invert the rest afterwards. 
data = ert.load("P:\\11208018-016-urban-geophysics\\4. Data\\ERT\\T4-Cold_right\\lin4_0000.pyg") # Load previously made pyg files. This one is for making the mesh. 
# pg.viewer.mpl.showDataContainerAsMatrix(data, "a", "m", "rhoa",cMin=100,cMax=5000,cmap='rainbow')
data["k"] = ert.geometricFactors(data)  # calculates configuration factors for ert data, this one has someting to do with topography
data["err"] = ert.estimateError(data) # calculates configuration factors for ert data

# pg.plt.plot(pg.x(data), pg.y(data), "x")
# %%

mesh = makeMesh(data.sensors()) # Make the mesh with electrode positions. 

# mesh.exportVTK("mesh3.vtk")
# %%

# I dont think this is used anywhere
#invmesh = mesh.createSubMesh(mesh.cells(mesh.cellMarkers() == 2))

# %%
# data.removeSensorIdx(166)
# data.removeSensorIdx(81)
# data.remove((data["a"]==166)*(data["b"]==166)*(data["m"]==166)*(data["n"]==166))
# data.remove((data["a"]==81)*(data["b"]==81)*(data["m"]==81)*(data["n"]==81))

#test inversion?
# mgr = ert.ERTManager(data, verbose=True)
# mgr.invert(mesh=mesh)
# mgr.saveResult()


import glob

yt=glob.glob('P:\\11208018-016-urban-geophysics\\4. Data\\ERT\\T4-Cold_right\\*.pyg')


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


# I dont think that this is used either
a1=np.array(mgr.inv.dataVals) # this is the actual raw data
# a1=np.array(mgr.inv.response) # this is the inversion response
# a1=np.array(mgr.fw.dataVals) #alsoas above
a2=np.array(mgr.fw.response) # also as above
aer=np.array(mgr.fw.dataErrs)
aer2=np.array(mgr.inv.errorVals)

mgr.showFit()
mgr.showResultAndFit()
# test=data['valid']
# ix=np.where((a1/a2>1.8) | (a1/a2<0.2))[0]

# for i in range(0,len(ix)):
#     data['valid'][ix[i]]=0
lala=np.c_[np.array(data['a']),np.array(data['b']),np.array(data['m']),np.array(data['n']),np.array(data['rhoa'])]  #no clue, but again not used

# plt.subplot(2,1,1)
# plt.plot(a1/a2)
# plt.subplot(2,1,2)
# plt.plot(aer)
# data.markInvalid(ix)
# data.removeInvalid()


# mgr = ert.ERTManager(data, verbose=True)
# mgr.invert(mesh=mesh)
# mgr.saveResult()
# mgr.showFit()


# mgr.showResultAndFit()
# modelPD = mgr.paraModel(model)  # do the mapping
# pg.show(mgr.paraDomain, modelPD, label='Model', cMap='Spectral_r',
#         logScale=True, cMin=25, cMax=150)