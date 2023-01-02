# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 13:49:04 2022

@author: obandohe
"""

import pygimli as pg
from pygimli.physics import TravelTimeManager
from pygimli.physics import traveltime
import os
import glob

import pyvista as pv

path = r'C:\Users\obandohe\OneDrive - Stichting Deltares\Documents\DELTARES PROJECTS_2021\012_WarmingUp4B\03_PROCESSING\002_PROCESSING_GEODE\SLEDGEHAMMER_GEODE\001_REFRACTION SEISMIC\04_PG_INVERSION\20220122-14.02\TravelTimeManager'
os.chdir(path)  


filename = glob.glob("*.vtk")

#%%

pl = pv.Plotter(window_size=(1600,1008))

pl.set_background('white', top='white')

vel = pv.read(filename[0])
vel.rename_array('Velocity', 'Vp [m/s]')

labels = dict(xlabel='Distance [m]', ylabel='Depth [m]')

#pl.subplot(0)

sargs = dict(height=0.05, vertical=False, position_x=0.13, position_y=0.2,color='black')


vel_clip = (vel.clip_box([-60,-20,-45,0,0,1])).clip_box([100,150,-45,0,0,1])

pl.add_mesh(vel_clip,scalars="Vp [m/s]",cmap='jet',clim=[200,2000],log_scale=True,scalar_bar_args=sargs)

pl.view_xy()
pl.show_bounds(bounds=[-20,100,-45,0,0,0],color='black',font_size=14,**labels)
pl.camera.Zoom(1.2)
pl.show()
