# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:05:02 2018

@author: karaouli
"""

import numpy as np



def make_pyg_3d(filename,coords,data):
    file=open(filename,'w')
    file.write("%d\n"%(coords.shape[0]))
    file.write('# x y z\n')
    
    for i in range(0,coords.shape[0]):
        file.write('%.2f  %.2f  %.2f\n'%(coords[i,0],coords[i,1],coords[i,2])) 

    file.write('%d\n'%(data.shape[0]))
    
    file.write('#a	b	m	n	rhoa	err\n')
    for i in range(0,data.shape[0]):
        buf="%d  %d   %d  %d  %.6f  %.2f\n"%(data[i,0],data[i,1],data[i,2],data[i,3],data[i,4],1)
        file.write(buf)
    file.write('0\n')
    file.close()