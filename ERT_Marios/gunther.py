import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
import numpy as np
import matplotlib.pyplot as plt

def makeMesh(pos, invbound=2, bound=10):
    """Generate mesh around electrodes."""
    xmin, xmax = min(pg.x(pos)), max(pg.x(pos))
    ymin, ymax = min(pg.z(pos)), max(pg.z(pos))
    print(xmin, xmax, ymin, ymax)
    xmid = (xmin+xmax)/2
    ymid = (ymin+ymax)/2
    world = mt.createWorld(start=[0, -2.0],
                           end=[7.75, 0])
    # world.translate([xmid, ymid, 0])  # some bug in createWorld!

    maxdep = min(pg.z(pos)) - invbound
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
    pos_xy=np.array(pos)
    pos_xy=np.c_[pos_xy[:,0],pos_xy[:,1]]
    geom = world 
    for po in pos_xy:
        geom.createNode(po, marker=-99)
        geom.createNode([po[0],po[1]-0.005], marker=-99)

        # geom.createNode(po - pg.Pos(0, 0.005))  # refinement

    # geom.exportPLC("mesh.poly")  # tetgen
    # geom.exportVTK("geom.vtk")  # vtk
    mesh = mt.createMesh(geom, quality=1.2,area=0.005)
    return mesh


data = ert.load("P:\\11208018-016-urban-geophysics\\4. Data\\ERT\\T4-Cold_right\\lin4_0000.pyg")
# pg.viewer.mpl.showDataContainerAsMatrix(data, "a", "m", "rhoa",cMin=100,cMax=5000,cmap='rainbow')
print(data['rhoa'])

    
data["k"] = ert.geometricFactors(data)
print(min(data["k"]), max(data["k"]))
data["err"] = ert.estimateError(data)
print(min(data["err"]), max(data["err"]))
# pg.plt.plot(pg.x(data), pg.y(data), "x")
# %%
mesh = makeMesh(data.sensors())
# print(mesh)
# mesh.exportVTK("mesh3.vtk")
# %%


invmesh = mesh.createSubMesh(mesh.cells(mesh.cellMarkers() == 2))
print(invmesh)
# %%
# data.removeSensorIdx(166)
# data.removeSensorIdx(81)
# data.remove((data["a"]==166)*(data["b"]==166)*(data["m"]==166)*(data["n"]==166))
# data.remove((data["a"]==81)*(data["b"]==81)*(data["m"]==81)*(data["n"]==81))

mgr = ert.ERTManager(data, verbose=True)
mgr.invert(mesh=mesh)
mgr.saveResult()
#

import glob

yt=glob.glob('P:\\11208018-016-urban-geophysics\\4. Data\\ERT\\T4-Cold_right\\*.pyg')


for i in range(0,len(yt)):
    data = ert.load(yt[i])
    data["k"] = ert.geometricFactors(data)
    data["err"] = ert.estimateError(data)
    mgr = ert.ERTManager(data, verbose=True)
    mgr.invert(mesh=mesh,maxIter=5)
    mgr.saveResult(folder=yt[i][:-4])
    np.savetxt('pgr.txt',[i])
    plt.close()
    # mgr.showFit()


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
lala=np.c_[np.array(data['a']),np.array(data['b']),np.array(data['m']),np.array(data['n']),np.array(data['rhoa'])]

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