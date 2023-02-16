# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 13:47:38 2023

@author: leentvaa
"""
import numpy as np
def make_schd(filename, elec, quad, abem):
    """

    Parameters
    ----------
    filename : .txt
        Schedule file for MPT.
    elec : Array
        Electrode positions: x,y,z
    quad : Array
        ABMN configuratie.
    abem : BOOL
        If ABEM device is used, choose abem = True, if not, abem = False.

    Returns
    -------
    None.

    """
    
    
    f = open(filename, "w")
    
    f.write('#ERTLab_Solver\n')
    f.write('#verison_number= 1.3.0\n')
    f.write('#date= 15-Oct-14\n')
    f.write('!Electrodes input format\n')
    f.write('#elec_no_cable= 1\n')
    f.write('#elec_cable_col= 1\n')
    f.write('#elec_id_col= 2\n')
    f.write('#elec_x_col= 3\n')
    f.write('#elec_y_col= 4\n')
    f.write('#elec_z_col= 5\n')
    f.write('#elec_elev_col= -1\n')
    f.write('#elec_type_col= -1\n')
    f.write('!Data input format\n')
    f.write('#data_id_col= 1\n')
    f.write('#data_a_cable_col= 2\n')
    f.write('#data_a_elec_col= 3\n')
    f.write('#data_b_cable_col= 4\n')
    f.write('#data_b_elec_col= 5\n')
    f.write('#data_m_cable_col= 6\n')
    f.write('#data_m_elec_col= 7\n')
    f.write('#data_n_cable_col= 8\n')
    f.write('#data_n_elec_col= 9\n')
    f.write('#elec_start\n')
    f.write('! Cable	Elec	X	Y	Z\n')
    
    for i in range(0,len(elec)):
        f.write(str(1)+' '+str(i)+' '+str(elec[i,0])+' '+str(elec[i,1])+' '+str(elec[i,2])+'\n')  #vraag aan bas hoe ik dit mooi maak
    
    f.write('#elec_end\n')
    f.write('#trans_start\n')
    f.write('! PIN	Well/Line	Electrode\n')
    
    pin_abem = np.concatenate((np.linspace(11,30,20), np.linspace(1,10,11), np.linspace(31,61,21)))
    if abem == True:
        for j in range(1, len(elec)):
            f.write(str(i))
            f.write(str(1))
            f.write(str(i)+'\n')
   
    else:
        for k in range(1, len(elec)):
            f.write(str(i))
            f.write(str(1))
            f.write(str(pin_abem[i])+'\n')

            
    f.write('#trans_end\n')
    f.write('#data_start\n')
    f.write('! ID	Cable_A	A	Cable_B	B	Cable_M	M	Cable_N	N\n')
    
    for i in range (0, len(quad)):
        f.write(str(i))
        f.write(str(1))
        f.write(str(quad[i,0]))
        f.write(str(1))
        f.write(str(quad[i,1]))
        f.write(str(1))
        f.write(str(quad[i,2]))
        f.write(str(1))
        f.write(str(quad[i,3])+'\n')

        
    f.write('#data_end\n')
    f.close()        
    return

D = 0.05
ELEC = np.array([
            [2., 0., -D],
            [2.1, 0., -D],
            [2.2, 0., -D],
            [2.3, 0., -D],
            [2.4, 0., -D],
            [2.5, 0., -D],
            [2.7, 0., -D],
            [0. ,0. ,0.],
            [0. ,0. ,0.],
            [0. ,0. ,0.],
            [3.7, 0., -0.5],
            [4.7, 0., -0.5],
            [5.7, 0., -0.5],
            [6.7, 0., -0.5]])

# Import file for ABMN configuratie
import glob
import sys
from read_mpt_data import read_mpt_data

path = "D:\\Foil2022\\Foil detection 5 cm\\"
yt = glob.glob(path+'*data')

for i in range(0,len(yt)):
    data, elec_data = read_mpt_data(yt[i])

# ABMN COORDS    
COORDS = np.c_[ELEC[data[:,0].astype('int32')-1],
           ELEC[data[:,1].astype('int32')-1],
           ELEC[data[:,2].astype('int32')-1],
           ELEC[data[:,3].astype('int32')-1]]
a = make_schd('testblabla.txt',ELEC,COORDS,1)