import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
import numpy as np
import matplotlib.pyplot as plt

def makeMesh(pos, invbound=2, bound=100):
    """Generate mesh around electrodes."""
    # xmin, xmax = min(pg.x(pos)), max(pg.x(pos))
    # ymin, ymax = min(pg.y(pos)), max(pg.y(pos))
    # print(xmin, xmax, ymin, ymax)
    # xmid = (xmin+xmax)/2
    # ymid = (ymin+ymax)/2
    # world = mt.createWorld(start=[-bound, -bound, -bound],
                           # end=[bound, bound, 0])
    # world.translate([xmid, ymid, 0])  # some bug in createWorld!

    # maxdep = min(pg.z(pos)) - invbound
    # sx = xmax - xmin + invbound * 2
    # sy = ymax - ymin + invbound * 2
    box = mt.createCube(pos=[8.75/2, 5.4/2, -1.25],
                        size=[8.75, 5.4, -2.5], boundaryMarker=1,marker=2, area=0.1)
    geom =  box
    for po in pos:
        geom.createNode(po, marker=-99)
        # geom.createNode(po - pg.Pos(0, 0, 0.05))  # refinement
        geom.createNode(po - pg.Pos(0, 0, -0.05))  # refinement
        geom.createNode(po - pg.Pos(0.05, 0, 0))  # refinement
        geom.createNode(po - pg.Pos(-0.05, 0 ,0))  # refinement
        geom.createNode(po - pg.Pos(0, 0.05 ,0))  # refinement
        geom.createNode(po - pg.Pos(0, -0.05 ,0))  # refinement


    #pivot electrodes
    geom.createNode([8.75/2, 5.4/2, -1.25], marker=-999)
    #refinmemt
    geom.createNode([8.75/2, 5.4/2, -1.25-0.05], marker=-999)

    #calibvation
    geom.createNode([0.25,0.25 , 0.0], marker=-1000)
    #refinemnet
    geom.createNode([0.25,0.25 , -0.05], marker=-1000)

    # geom.exportPLC("mesh.poly")  # tetgen
    # geom.exportVTK("geom.vtk")  # vtk
    mesh = mt.createMesh(geom, quality=1.3,area=0.1)
    return mesh


data = ert.load("P:\\11208018-016-urban-geophysics\\4. Data\\ERT\\T4-Cold_right\\pyg\\lin4_0000.pyg")
# pg.viewer.mpl.showDataContainerAsMatrix(data, "a", "m", "rhoa")
data["k"] = ert.geometricFactors(data)
print(min(data["k"]), max(data["k"]))
# data["err"] = ert.estimateError(data)
print(min(data["err"]), max(data["err"]))
print(data)
# pg.plt.plot(pg.x(data), pg.y(data), "x")
# %%
mesh = makeMesh(data.sensors())
# print(mesh)
# mesh.exportVTK("mesh35.vtk")
# 3D met finite domain 

hom = ert.simulate(mesh, res=100.0, scheme=data, sr=False,
                   calcOnly=True, verbose=True)

hom.save('homogeneous.ohm', 'a b m n u')


# gf=np.array(100.0/ (hom('u') / hom('i')))
# gf2=np.load('gf.npy')
# skata=gf/gf2
data.set('k', 100.0/ (hom('u') / hom('i')))
data.set('rhoa', data('k') * data('u') / 1)

data.save('3d__0014_rhoa.pyg', 'a b m n rhoa k u i')








# %%
invmesh = mesh.createSubMesh(mesh.cells(mesh.cellMarkers() == 2))
print(invmesh)
# %%
# data.removeSensorIdx(166)
# data.removeSensorIdx(81)
# data.remove((data["a"]==166)*(data["b"]==166)*(data["m"]==166)*(data["n"]==166))
# data.remove((data["a"]==81)*(data["b"]==81)*(data["m"]==81)*(data["n"]==81))

data = ert.load("3d__0014_rhoa.pyg")
data["err"] = ert.estimateError(data)

mgr = ert.ERTManager(data, verbose=True)
mgr.invert(mesh=mesh,maxIter=8)
mgr.saveResult(folder='trial_3d')
#

# import glob

# yt=glob.glob('c*.pyg')

# for i in range(500,len(yt)):
#     data = ert.load(yt[i])
    # data["k"] = ert.geometricFactors(data)
    # data["err"] = ert.estimateError(data)
    # # pg.viewer.mpl.showDataContainerAsMatrix(data, "a", "m", "rhoa",cMap='gist_rainbow')

    # mgr = ert.ERTManager(data, verbose=True)
    # mgr.invert(mesh=mesh,maxIter=5)
    # mgr.saveResult(folder=yt[i][:-4])
    # np.save('pgr.txt',i)
    # mgr.showFit()

# a1=np.array(mgr.inv.dataVals)
# a1=np.array(mgr.inv.response)
# a1=np.array(mgr.fw.dataVals)
# a2=np.array(mgr.fw.response)
# aer=np.array(mgr.fw.dataErrs)
# aer2=np.array(mgr.inv.errorVals)

# test=data['valid']
# ix=np.where((a1/a2>1.2) | (a1/a2<0.8))[0]

# for i in range(0,len(ix)):
#     data['valid'][ix[i]]=0


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