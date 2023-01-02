# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 23:04:18 2019

@author: karaouli
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 21:17:28 2019

@author: karaouli
"""
import numpy as np
from io import StringIO
import glob


def read_mpt_data(filename):

    file=open(filename,'r')
    lines_all=file.readlines()
    file.close()
    first_elec=0
    first_meas=0

    for i in range(0,len(lines_all)):
        lines = lines_all[i]
        if lines[0:11]=='#elec_start':
            k=i
            k=k+1
            lines = lines_all[k]  # get header
            k=k+1
            lines = lines_all[k]
            
            while lines[0:9]!='#elec_end':
                tmp=lines[7:].replace(',','.')
                lines=lines[:7]+tmp
          
                tmp=lines.replace(',',' ')

                tmp=np.loadtxt(StringIO(tmp))
         
                if first_elec==0:
                    elec=np.c_[tmp[1],tmp[2],tmp[3],tmp[4]]
                    first_elec=1
                else:
                    elec=np.r_[elec,np.c_[tmp[1],tmp[2],tmp[3],tmp[4]]]
                
                k=k+1
                lines=lines_all[k]
                
        elif lines[0:11]=='#data_start':
            k=i
            k=k+1
            lines = lines_all[k]  # get header
            k=k+1 
            lines = lines_all[k]  # get header
            k=k+1 
            lines = lines_all[k]  # get header
                   
            
            while (lines[0:9]!='#data_end') &  (lines[0:9]!='Run Compl'):
                   
                   tmp=lines[35:].replace(',','.')
                   lines=lines[:35]+tmp
                   tmp=lines.replace(',',' ')
                   tmp=tmp.replace('CH','00')
                   tmp=tmp.replace('GN','00')
                   
                   tmp=tmp.replace('*','00')
                   tmp=tmp.replace('TX','00')
                   tmp=tmp.replace('Resist.','00')
                   tmp=tmp.replace('out','00')
                   tmp=tmp.replace('of','00')
                   tmp=tmp.replace('range','00')
                   
                   tmp=tmp.replace('Error_Zero_Current','00')
                   tmp=tmp.replace('Raw_Voltages:','00')
                   
                   tmp = tmp.replace('_','.')
              
                   tmp=np.loadtxt(StringIO(tmp))
                   
                   if len(tmp)<23:
                       add=22-len(tmp)
                       tmp=np.r_[tmp,np.zeros((add,))]
                   # print(tmp)
                   if first_meas==0:
                       meas=np.c_[tmp[2],tmp[4],tmp[6],tmp[8],tmp[9],tmp[10],tmp[11],tmp[14],tmp[15],tmp[18]]
                       first_meas=1
                   else:
                       meas=np.r_[meas,np.c_[tmp[2],tmp[4],tmp[6],tmp[8],tmp[9],tmp[10],tmp[11],tmp[14],tmp[15],tmp[18]]]
                   
                   k=k+1
                   lines=lines_all[k]
                       
                
       
           
#    ix=np.where(meas[:,4]==0)[0]    
#    meas=np.delete(meas,ix,axis=0) 
        
    return meas,elec    
        
    
