# Import packages

from obspy.core import read
import numpy as np
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
import os
from obspy.signal.trigger import classic_sta_lta
from obspy.core import read
from obspy.signal.trigger import plot_trigger
from obspy.signal.trigger import recursive_sta_lta
from obspy.signal.trigger import pk_baer
import glob
import csv
import pandas as pd

#%% 

def write_vs(filename,dist,tt,dx,n_sources,source_pos):

    file = open(filename ,'w')
    file.write('1996 0 3.000000\n0 %d %f\n'%(n_sources,dx))
    
    #for i in range(0,len(freq)):
    file.write('%0.6f %d 0.000000\n'%(float(source_pos), len(tt)))  
        
    for i in range(0,len(dist)):    
        file.write('%0.6f %0.6f 1 \n'%(dist[i],tt[i]))
    file.write('0 0 \n 0 \n 0 0 \n')
    file.close()
    return

def Pick_first_breaks(file,f,files):
    #Plot Signal
    file.plot(type="relative")
    
    #Extract trace info  # loop over traces in file TOOOODDOOOO
    for iii in range(np.shape(file.traces)[0]):
        trace = file.traces[iii]#.filter('bandpass',freqmin = 10, freqmax=500, zerophase = True)
        
        #Extract sampling rate (too big in these files)
        df = trace.stats.sampling_rate/100
        
        #Plot calculated trigger start
       # cft = recursive_sta_lta(trace, int(5 * df), int(10 * df))
       # plot_trigger(trace, cft, 1.7, 0.5)
        
        #Find locatin of trigger start in array
        p_pick, phase_info = pk_baer(trace, df, 20, 60, 7.0, 12.0, 100, 100)
        if np.size(p_pick) > 1:
            p_pick = p_pick[0]
            
        file.traces[iii].data = file.traces[iii].data[int(p_pick):]
            

    # Write output file
    #outfile = write_vs(path+"FBpicked"+str(f)+".vs",np.linspace(0,1,np.shape(file.traces)[0]),file.traces[0].data,1,4,file.traces[0].stats.seg2.SOURCE_LOCATION)
    
    outfile = open(path+"FBpicked"+str(f)+".csv",'w') #two colomn file
    writer = csv.writer(outfile)
    
    writer.writerow(file.traces)
    outfile.close()
        
    return 


#%%

#Import Files
path = 'C:\\Users\\leentvaa\\PythonScripts\\Refraction_Seismic\\'
files = glob.glob(path+'*.dat')

for f in range(len(files)): 
    file = read(files[f]) #find a way to put this part in the def or another def 
    Result = Pick_first_breaks(file,f,files)

#%%

#plot cutted part weghalen later
'''
new = pd.read_csv('')
with open('C:\\Users\\leentvaa\\PythonScripts\\Refraction_Seismic\\FBPicked1.csv', 'r') as file:
    reader = csv.reader(file)
reader.plot(type="relative")
'''