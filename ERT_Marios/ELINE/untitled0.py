# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 13:38:10 2023

@author: leentvaa

Write a function that creates a pyg file from MPT/ABEM



# Start with MPT 


rough idea of setup:
    def read data , input is abem of mpt
    def calc gf
    def elec positions
    def write file
    

"""

# Import packages

import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import os

# Pygimli
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert

# Local
from make_pyg import make_pyg
from geofac2 import geofac2
from read_mpt_data import read_mpt_data

var low  , 
def read_raw_data(filedir, hardware='MPT', custom_elec=0):

    VI = {}
    AR = {}
    AR_recalc = {}

    for filename in os.listdir(filedir):
        if hardware == 'MPT':
            if filename.endswith('.Data'):
                DATA, ELEC = read_mpt_data(filedir+filename)

                if np.size(custom_elec) != 0:
                    ELEC = custom_elec

                # Coordinates Electrodes [Ax Ay Az Bx By Bz Mx My Mz Nx Ny Nz]
                COORDS = np.c_[ELEC[DATA[:, 0].astype('int32')-1],
                               ELEC[DATA[:, 1].astype('int32')-1],
                               ELEC[DATA[:, 2].astype('int32')-1],
                               ELEC[DATA[:, 3].astype('int32')-1]]

                # Geometrical factor
                GF, GF2 = geofac2(COORDS)

                # Resistance V\I
                VI[filename] = DATA[:, 5]

                # Apparent Resistivity
                AR[filename] = DATA[:, 4]
                AR_recalc[filename] = VI[filename]*GF

    return DATA, ELEC, VI, AR, AR_recalc, GF

#%%
from pathlib import Path

class MptFile:
    
    def __init__(self, filepath, sep=' '):
        self.sep = sep
        self.__open_file(filepath)
    
    def __open_file(self, filepath):
        with open(filepath, 'r') as f:
            self.lines = f.readlines()
    
    @property
    def elec_data(self):
        idx_begin = self.lines.index('#elec_start\n') + 1
        idx_end = self.lines.index('#elec_end\n')
        elec_data = self.lines[idx_begin:idx_end]
        
        columns = elec_data[0].strip('\n')
        data = pd.Series(elec_data[1:]).str.strip('\n')
        
        data = data.str.split(self.sep, expand=True)
        # data.columns = columns.split(self.sep)
        
        return data
        


filedir = Path(r'P:/11208018-016-urban-geophysics/4. Data/ERT/T3-Warm_left')

mpt_files = filedir.glob('*.Data')

for f in mpt_files:
    # name, ext = os.path.splitext(os.path.basename(f))
    name = f.stem
    
    mpt = MptFile(f)    
    data, elec = read_mpt_data(f)
    break

#%%


elecnew = np.array([[0., 0., 0.],
                    [0.25, 0., 0.],
                    [0.5, 0., 0.],
                    [0.75, 0., 0.],
                    [1., 0., 0.],
                    [1.25, 0., 0.],
                    [1.5, 0., 0.],
                    [1.75, 0., 0.],
                    [2., 0., 0.],
                    [2.25, 0., 0.],
                    [2.5, 0., 0.],
                    [2.75, 0., 0.],
                    [3., 0., 0.],
                    [3.25, 0., 0.],
                    [3.5, 0., 0.],
                    [3.75, 0., 0.],
                    [4., 0., 0.],
                    [4.25, 0., 0.],
                    [4.5, 0., 0.],
                    [4.75, 0., 0.],
                    [5., 0., 0.],
                    [5.25, 0., 0.],
                    [5.5, 0., 0.],
                    [5.75, 0., 0.],
                    [6., 0., 0.],
                    [6.25, 0., 0.],
                    [6.5, 0., 0.],
                    [6.75, 0., 0.],
                    [7., 0., 0.],
                    [7.25, 0., 0.],
                    [7.5, 0., 0.],
                    [7.75, 0., 0.],
                    [0., 0.5, 0.],
                    [0.25, 0.5, 0.],
                    [0.5, 0.5, 0.],
                    [0.75, 0.5, 0.],
                    [1., 0.5, 0.],
                    [1.25, 0.5, 0.],
                    [1.5, 0.5, 0.],
                    [1.75, 0.5, 0.],
                    [2., 0.5, 0.],
                    [2.25, 0.5, 0.],
                    [2.5, 0.5, 0.],
                    [2.75, 0.5, 0.],
                    [3., 0.5, 0.],
                    [3.25, 0.5, 0.],
                    [3.5, 0.5, 0.],
                    [3.75, 0.5, 0.],
                    [4., 0.5, 0.],
                    [4.25, 0.5, 0.],
                    [4.5, 0.5, 0.],
                    [4.75, 0.5, 0.],
                    [5., 0.5, 0.],
                    [5.25, 0.5, 0.],
                    [5.5, 0.5, 0.],
                    [5.75, 0.5, 0.],
                    [6., 0.5, 0.],
                    [6.25, 0.5, 0.],
                    [6.5, 0.5, 0.],
                    [6.75, 0.5, 0.],
                    [7., 0.5, 0.],
                    [7.25, 0.5, 0.],
                    [7.5, 0.5, 0.],
                    [7.75, 1., 0.],
                    [0., 1., 0.],
                    [0.25, 1., 0.],
                    [0.5, 1., 0.],
                    [0.75, 1., 0.],
                    [1., 1., 0.],
                    [1.25, 1., 0.],
                    [1.5, 1., 0.],
                    [1.75, 1., 0.],
                    [2., 1., 0.],
                    [2.25, 1., 0.],
                    [2.5, 1., 0.],
                    [2.75, 1., 0.],
                    [3., 1., 0.],
                    [3.25, 1., 0.],
                    [3.5, 1., 0.],
                    [3.75, 1., 0.],
                    [4., 1., 0.],
                    [4.25, 1., 0.],
                    [4.5, 1., 0.],
                    [4.75, 1., 0.],
                    [5., 1., 0.],
                    [5.25, 1., 0.],
                    [5.5, 1., 0.],
                    [5.75, 1., 0.],
                    [6., 1., 0.],
                    [6.25, 1., 0.],
                    [6.5, 1., 0.],
                    [6.75, 1., 0.],
                    [7., 1., 0.],
                    [7.25, 1., 0.],
                    [7.5, 1., 0.],
                    [7.75, 1., 0.],
                    [0., 1.5, 0.],
                    [0.25, 1.5, 0.],
                    [0.5, 1.5, 0.],
                    [0.75, 1.5, 0.],
                    [1., 1.5, 0.],
                    [1.25, 1.5, 0.],
                    [1.5, 1.5, 0.],
                    [1.75, 1.5, 0.],
                    [2., 1.5, 0.],
                    [2.25, 1.5, 0.],
                    [2.5, 1.5, 0.],
                    [2.75, 1.5, 0.],
                    [3., 1.5, 0.],
                    [3.25, 1.5, 0.],
                    [3.5, 1.5, 0.],
                    [3.75, 1.5, 0.],
                    [4., 1.5, 0.],
                    [4.25, 1.5, 0.],
                    [4.5, 1.5, 0.],
                    [4.75, 1.5, 0.],
                    [5., 1.5, 0.],
                    [5.25, 1.5, 0.],
                    [5.5, 1.5, 0.],
                    [5.75, 1.5, 0.],
                    [6., 1.5, 0.],
                    [6.25, 1.5, 0.],
                    [6.5, 1.5, 0.],
                    [6.75, 1.5, 0.],
                    [7., 1.5, 0.],
                    [7.25, 1.5, 0.],
                    [7.5, 1.5, 0.],
                    [7.75, 1.5, 0.]])
DATA, ELEC, VI, AR, AR_recalc, GF = read_raw_data(
    "P:/11208018-016-urban-geophysics/4. Data/ERT/T3-Warm_left/", custom_elec=elecnew)

print(AR_recalc["LAB_BRICK_SHORT220221124_1636.Data"],
      AR["LAB_BRICK_SHORT220221124_1636.Data"])


