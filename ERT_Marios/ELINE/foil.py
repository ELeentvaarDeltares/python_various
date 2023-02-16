import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def makeMesh(pos, invbound=2, bound=10, line = 0):
    """Generate mesh around electrodes."""
    
    if line > 0:
        layer = - line*0.1
        layer2 = layer + 0.01
    else:
        layer = -9.98
        layer2 = -9.99
        
    world = mt.createWorld(start=[0, -10],
                           end=[8.75, 0], layers = [layer,layer2],worldMarker = [0,1,2])
  
    pos_xy=np.array(pos)
    pos_xy=np.c_[pos_xy[:,0],pos_xy[:,2]] # array with positions of only x and y coordinates electrodes

    geom = world
    for po in pos_xy:
        geom.createNode(po, marker=-99)
        geom.createNode([po[0],po[1]-0.005], marker=-99)
    
    #geom.exportVTK("geom"+str(line)+".vtk")  # vtk
    mesh = mt.createMesh(geom, quality=1.2,area=0.01)
    #pg.show(geom)
    
    return mesh



# Calculate GF
# Import packages
from geofac2 import geofac2
import glob 
from read_mpt_data import read_mpt_data

# Electrode positions
D = 0.5
ELEC = np.array([
            [2., 0., -D],
            [2.1, 0., -D],
            [2.2, 0., -D],
            [2.3, 0., -D],
            [2.4, 0., -D],
            [2.5, 0., -D],
            [2.7, 0., -D],
            [0.1 ,0 ,0.],
            [0.2 ,0 ,0.],
            [0.3 ,0 ,0.],
            [3.7, 0., -0.5],
            [4.7, 0., -0.5],
            [5.7, 0., -0.5],
            [6.7, 0., -0.5],
            [0.4 ,0 ,0.]])

ELEC2D = np.array([
            [2.,  -D],
            [2.1,  -D],
            [2.2,  -D],
            [2.3,  -D],
            [2.4,  -D],
            [2.5,  -D],
            [2.7,  -D],
            [0. ,0.],
            [0. ,0.],
            [0. ,0.],
            [3.7,  -0.5],
            [4.7,  -0.5],
            [5.7,  -0.5],
            [6.7,  -0.5],
           ])


# Import file for ABMN configuratie

path = "D:\\Foil2022\\Foil detection 50 cm\\"
yt = glob.glob(path+'*data')

for i in range(0,len(yt)):
    print(yt[i])
    data, elec_data = read_mpt_data(yt[i])
    if i==0:
        out=data[:,5]
    else:
        out=np.c_[out,data[:,5]]

# ABMN COORDS    

COORDS = np.c_[ELEC[data[:,0].astype('int32')-1],
           ELEC[data[:,1].astype('int32')-1],
           ELEC[data[:,2].astype('int32')-1],
           ELEC[data[:,3].astype('int32')-1]]


# Calculate GF
GF, GF2 = geofac2(COORDS)



for k in range(out.shape[1]):
    out[:,k] = out[:,k] * GF

import sys
sys.path.insert(0,'C:/Users/leentvaa/python_various/ERT_Marios/MARIOS')
from make_pyg_3d import make_pyg_3d
import scipy 

for i in range(0,out.shape[1]):
    iiii=np.where(out[:,i]>-np.inf)[0]
    tt=np.c_[data[iiii,:4],out[iiii,i]]
    make_pyg_3d(path+'lin1_%04d.pyg'%i, ELEC, tt)
    
scheme = pg.DataContainerERT(path+'lin1_0000.pyg') 
scheme_plot = scipy.ndimage.gaussian_filter(scheme['rhoa'], 5)


# Kernel Calculation

for line in range(0,30):

    mesh = makeMesh(ELEC,line=line) # Make the mesh with electrode positions
    if line == 0 :
        kMap = [[0,300], [3,300]]
    else:
        kMap = [[2,300],[0,300000],[3,300000]]
    
    K = pg.solver.parseMapToCellArray(kMap, mesh)
    #pg.show(mesh, data=K, label=pg.unit('res'), showMesh=False)
    #plt.show()
    SIM = ert.simulate(mesh,scheme=scheme, res = K,  sr=False,
                     calcOnly=True, verbose=False)

    SIM_plot = scipy.ndimage.gaussian_filter(GF*SIM['u']/SIM['i'],2)
    plt.plot(SIM_plot)
    plt.plot(scheme_plot)
    plt.axvline(15)
    plt.title('foildepth = '+str(line*0.1))
    #plt.plot(np.linspace(0,7.75,len(scheme_plot)),scheme_plot-SIM_plot)

plt.show()
   # DATA = SIM['u']*GF
    #plt.plot(DATA)






















''' 
#%%

data = ert.load("E:\Foil2022\Foil detection 5 cm\lin1_0000.pyg") # Load previously made pyg files. This one is for making the mesh.

new_elec = np.array([[2., 0., -0.70],
            [2.1, 0., -0.05],
            [2.2, 0., -0.05],
            [2.3, 0., -0.05],
            [2.4, 0., -0.05],
            [2.5, 0., -0.05],
            [2.7, 0., -0.05],
            [0. ,0. ,0.],
            [0. ,0. ,0.],
            [0. ,0. ,0.],
            [3.7, 0., -0.5],
            [4.7, 0., -0.5],
            [5.7, 0., -0.5],
            [6.7, 0., -0.5]])

scheme = ert.createData(elecs=new_elec, schemeName='pd')

coor=np.c_[new_elec[data[:,0].astype('int32')-1],
           new_elec[data[:,1].astype('int32')-1],
           new_elec[data[:,2].astype('int32')-1],
           new_elec[data[:,3].astype('int32')-1]]
from geofac2 import geofac2 
gf = geofac2(coor)
print(gf)

for line in range(0,10):
    mesh = makeMesh(scheme.sensors(),line=line) # Make the mesh with electrode positions
    if line == 0 :
        kMap = [3,1/300]
    else:
        kMap = [[2,1/300],[0,1/30000], [3,1/300]]
    K = pg.solver.parseMapToCellArray(kMap, mesh)
    #pg.show(mesh, data=K, label='Hydraulic conductivity $K$ in m$/$s', cMin=1e-5,
        #cMax=1e-2, logScale=True, grid=True)
    hom = ert.simulate(mesh, res=K, scheme=scheme, sr=False,
                       calcOnly=True, verbose=True)
    
    print(hom)
    # #hom[line].save('model'+str(line)+'.ohm', 'a b m n u')
    # scheme.set('k', 100.0/ (hom['u'] / hom['i']))
    # #print(scheme['k'], scheme['u'])
    # scheme.set('rhoa2', scheme('k') * hom('u') / 1)
    
    
    plt.plot(np.linspace(0,7.75,num=hom['u'][1:].shape[0]), hom['u'][1:])
    from scipy.ndimage.filters import gaussian_filter1d
    ysmoothed = gaussian_filter1d(hom['u'][1:], sigma=2)
    plt.plot(np.linspace(0,7.75,num=hom['u'][1:].shape[0]), ysmoothed)
    
plt.show()



#%%

import glob
from pygimli import Inversion
yt=glob.glob('E:\Foil2022\Foil detection 5 cm\*.pyg')


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



'''




