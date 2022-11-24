import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
import numpy as np
import matplotlib.pyplot as plt
import os

# 2D ERT Data inversion

def makeMesh(pos, invbound=2, bound=100):
    """Generate mesh around electrodes."""
    xmin, xmax = min(pg.x(pos)), max(pg.x(pos))
    ymin, ymax = min(pg.z(pos)), max(pg.z(pos))
    print(xmin, xmax, ymin, ymax)
    xmid = (xmin+xmax)/2
    ymid = (ymin+ymax)/2
    world = mt.createWorld(start=[0, -2.0],
                           end=[7.75, 0])

    maxdep = min(pg.z(pos)) - invbound
    sx = xmax - xmin + invbound * 2
    sy = ymax - ymin + invbound * 2
    verts=[[xmax,0],[xmin,0]]
    temp=np.array(pos)
    for i in range(0,len(temp)):
        verts.append([temp[i,0],temp[i,2]-0.1])
    box=mt.createPolygon(verts,isClosed=True,marker=2,area=0.1)
    pos_xy=np.array(pos)
    pos_xy=np.c_[pos_xy[:,0],pos_xy[:,1]]
    geom = world + box
    for po in pos_xy:
        geom.createNode(po, marker=-99)
        geom.createNode([po[0],po[1]-0.005], marker=-99)

    mesh = mt.createMesh(geom, quality=1.2,area=3)
    pg.show(mesh)
    return mesh


data = ert.load("lin1_0000.pyg")
pg.viewer.mpl.showDataContainerAsMatrix(data, "a", "m", "rhoa",cMin=1,cMax=20,cmap='rainbow')

data["k"] = ert.geometricFactors(data)
data["err"] = ert.estimateError(data)
mesh = makeMesh(data.sensors())


invmesh = mesh.createSubMesh(mesh.cells(mesh.cellMarkers() == 2))

def saveResult2(self,i, folder=None, size=(16, 10), **kwargs): #to save the results in a logical way to insert in salinity program easily. 
        subfolder = self.__class__.__name__
        path = os.getcwd()
        m = pg.Mesh(self.paraDomain)
        m['Resistivity'] = self.paraModel(self.model)
        m['Resistivity (log10)'] = np.log10(m['Resistivity'])
        m['Coverage'] = self.coverage()
        m['S_Coverage'] = self.standardizedCoverage()
        m.exportVTK(os.path.join(path, str('resistivity')+str(i)))
        return path

import glob

yt=glob.glob('*.pyg')
for i in range(0,len(yt)):
    data = ert.load(yt[i])
    data["k"] = ert.geometricFactors(data)
    data["err"] = ert.estimateError(data)
    mgr = ert.ERTManager(data, verbose=True)
    mgr.invert(mesh=mesh)
    mgr.saveResult()
    np.savetxt('pgr.txt',[i])
    plt.close()
    vtk = saveResult2(mgr, i) 


a1=np.array(mgr.inv.dataVals) # this is the actual raw data
a2=np.array(mgr.fw.response) # also as above
aer=np.array(mgr.fw.dataErrs)
aer2=np.array(mgr.inv.errorVals)

mgr.showFit()
mgr.showResultAndFit()
