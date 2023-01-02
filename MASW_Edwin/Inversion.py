# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 13:49:04 2022

@author: obandohe
"""

import pygimli as pg
from pygimli.physics import TravelTimeManager
from pygimli.physics import traveltime
import os

path = r'C:\Users\leentvaa\PythonScripts\Refraction_Seismic\to_Eline\to_Eline'
os.chdir(path)  

data = traveltime.load('srt_UG.csv')

#%%

fig, ax = pg.plt.subplots()
pg.physics.traveltime.drawFirstPicks(ax, data)

#%%

mgr = TravelTimeManager()

# Alternatively, one can plot a matrix plot of apparent velocities which is the
# more general function also making sense for crosshole data.
ax, cbar = mgr.showData(data)

#%%

mgr.invert(data, secNodes=3, paraMaxCellSize=0.01,
           zWeight=0.2, vTop=100, vBottom=2000,
           verbose=1)

#%%
ax, cbar = mgr.showResult(logScale=False,cmap='jet',cMin=100,cMax=400)
#mgr.drawRayPaths(ax=ax, color="w", lw=0.3, alpha=0.3)
ax.set_ylabel('Depth [m]')
ax.set_xlabel('Distance [m]')

ax.set_ylim(-2.0,0)
ax.set_xlim(0,7.0)

#%%

