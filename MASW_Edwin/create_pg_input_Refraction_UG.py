# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:54:49 2021

@author: obandohe
"""

def get_input_data(name_of_vs,dx):

    file=open(name_of_vs,'r')
    aa=file.readline()
    aa=file.readline()
    tmp=np.fromstring(aa,sep=' ')
    no_sources=tmp[1]
    data=np.zeros((1,3))
    for i in range(0,np.int32(no_sources)):
        aa=file.readline()
        tmp=np.fromstring(aa,sep=' ')
        xs=tmp[0]
        num=tmp[1]
        for j in range(0,np.int32(num)):
            aa=file.readline()
            tmp=np.fromstring(aa,sep=' ')
            if len(tmp)>4:
                data_tmp=np.c_[xs,tmp[0],tmp[3]]
            else:
                data_tmp=np.c_[xs,tmp[0],tmp[1]]   
            
             
            data=np.r_[data,data_tmp]
    
    file.close()    
    data=data[1:,:]
    tt = data[:,2]
    dist = data[:,1]*dx

    return(dist,tt)


def write_vs(filename,dist,tt,dx,n_sources,source_pos):

    file = open(filename ,'w')
    file.write('1996 0 3.000000\n0 %d %f\n'%(n_sources,dx))
    
    #for i in range(0,len(freq)):
    file.write('%0.6f %d 0.000000\n'%(float(source_pos), len(tt)))  
        
    for i in range(0,len(dist)):    
        file.write('%0.6f %0.6f 1 \n'%(dist[i],tt[i]))
    file.write('0 0 \n 0 \n 0 0 \n')
    file.close()
    
 

def create_pgInput_3D_full(xyz,n_sources,tt_sel,filname):

    import numpy as np
       
    aa = np.arange(1,76)
    bb = np.ones((len(aa),n_sources))
    cc = (bb.T*aa).T
    size =  np.shape(cc)
    dd = np.reshape(cc.T,(size[0]*size[1],1))
    bb = (np.ones((len(aa),n_sources)))*len(aa)+np.array(([ikk for ikk in range(0,n_sources)]))
    size =  np.shape(bb)
    dd_2 = np.reshape(bb.T,(size[0]*size[1],1))
    
    s = dd_2
    g = dd
    
    
    #% Make outpit file 
    file=open(filname,'w') 
    file.write('%d\n'%xyz.shape[0])   
    file.write('#x y z\n')
    for i in range(0,xyz.shape[0]):
        file.write('%.4f\t%.4f\t%.4f\n'%(xyz[i,0],xyz[i,1],-xyz[i,2]))
        
        
    file.write('%d\n'%s.shape[0])   
    #file.write('# g s err t valid\n')
    file.write('#s g t \n')
    
    # Notice that is receiver source pair
    for i in range(0,s.shape[0]):
        
        file.write('%d\t%.7f\t%.7f\n'%(s[i]+1,g[i],tt_sel[i]/1000))
        #file.write('%d\t%d\t%.7f\t%.7f\t%d\n'%(pair[i,0],pair[i,1],pair[i,2]/1000,1))             
    
    #file.write('0\n')
    
    file.close()



def model_wmup5c_3D(depth,source_depth):
    
    import numpy as np
    
    n_rec = 25

    z_rec = np.hstack((depth,depth,depth))
    x_P01 = np.zeros((n_rec))+1
    x_P02 = np.zeros((n_rec))
    x_P04 = np.zeros((n_rec))+1
    
    x_rec = np.hstack((x_P01,x_P02,x_P04)) 
    y_P01 = np.zeros((n_rec))+1
    y_P02 = np.zeros((n_rec))+1
    y_P04 = np.zeros((n_rec))
    y_rec = np.hstack((y_P01,y_P02,y_P04)) 
    
    
    all_coord = np.zeros((3*n_rec,3))
    all_coord[:,0] = x_rec
    all_coord[:,1] = y_rec
    all_coord[:,2] = z_rec
    
    
    coord_src = np.zeros((len(source_depth),3))
    coord_src[:,2] = source_depth
    
    xyz = np.vstack((all_coord,coord_src)) 


    return xyz
   

#% -----------------------------------------------------------------------------------------------
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


path = r'C:\\Users\\leentvaa\\PythonScripts\\Refraction_Seismic\\'
os.chdir(path)


filename = glob.glob('*.vs')


dx = 1.0

full_tt = np.zeros((59,17))

for ik in range(17):
    
    name_of_vs = filename[ik]    
    
    dist,tt = get_input_data(name_of_vs,dx)
    
    full_tt[:,ik] = tt



plt.plot(dist,full_tt,'o--',color='darkblue')
plt.xlabel('X - coordinates [m]')
plt.ylabel('Travel times [ms]')
plt.grid()



#%%
sources = np.array([-0.5,0,0.48,1.08,1.44,2.04,2.52,3.0,3.48,4.08,4.56,5.04,5.52,6.0,6.48,6.96])

x_coord = np.unique(np.hstack((dist,sources))) 

xyz = np.zeros((len(x_coord),3))
xyz[:,0] = x_coord

source_index = np.zeros((len(sources)))
ip = 0
while ip < len(sources):
    source_index[ip] = np.where(x_coord==sources[ip])[0]
    ip+=1




iq = 0

sel_tt = []
g = []
s = []


iq = 0

while iq < 16:

    sel_tt0 = np.delete(full_tt[:,iq],[int(source_index[iq]-1)])
    sel_tt = np.hstack((sel_tt,sel_tt0))
    
    g0 = np.delete(np.arange(1,60),[int(source_index[iq]-1)])
    g = np.hstack((g,g0))
    
    
    s0 = np.repeat([int(source_index[iq]+1)],58)
    s = np.hstack((s,s0))
    
    iq+=1



#%%

filname = 'srt_UG.csv'

#% Make outpit file 
file=open(filname,'w') 
file.write('%d\n'%xyz.shape[0])   
file.write('#x y z\n')
for i in range(0,xyz.shape[0]):
    file.write('%.4f\t%.4f\t%.4f\n'%(xyz[i,0],xyz[i,1],xyz[i,2]))
    
    
file.write('%d\n'%s.shape[0])   
#file.write('# g s err t valid\n')
file.write('#s g t \n')

# Notice that is receiver source pair
for i in range(0,s.shape[0]):
    
    file.write('%d\t%.7f\t%.7f\n'%(s[i],g[i]+1,sel_tt[i]/1000))
    #file.write('%d\t%d\t%.7f\t%.7f\t%d\n'%(pair[i,0],pair[i,1],pair[i,2]/1000,1))             

#file.write('0\n')

file.close()





#%%
# =============================================================================
# 
# #%%
# num_m = 20
# 
# all_names = {60:[('HW0'+str(num+1)+'/'+'HW0'+str(num+1)+'_60'+'*.vs' if num+1 <=9 else 'HW'+str(num+1)+'/'+'HW'+str(num+1)+'_60'+'*.vs' ) for num in range(0,num_m)],
#              100:[('HW0'+str(num+1)+'/'+'HW0'+str(num+1)+'_100'+'*.vs' if num+1 <=9 else 'HW'+str(num+1)+'/'+'HW'+str(num+1)+'_100'+'*.vs' ) for num in range(0,num_m)],
#              140:[('HW0'+str(num+1)+'/'+'HW0'+str(num+1)+'_140'+'*.vs' if num+1 <=9 else 'HW'+str(num+1)+'/'+'HW'+str(num+1)+'_140'+'*.vs' ) for num in range(0,num_m)],
#              180:[('HW0'+str(num+1)+'/'+'HW0'+str(num+1)+'_180'+'*.vs' if num+1 <=9 else 'HW'+str(num+1)+'/'+'HW'+str(num+1)+'_180'+'*.vs' ) for num in range(0,num_m)],
#              220:[('HW0'+str(num+1)+'/'+'HW0'+str(num+1)+'_220'+'*.vs' if num+1 <=9 else 'HW'+str(num+1)+'/'+'HW'+str(num+1)+'_220'+'*.vs' ) for num in range(0,num_m)]}
#  
# 
# all_names[60].append('T0_1/T01_60*.vs')
# all_names[100].append('T0_1/T01_100*.vs')
# all_names[140].append('T0_1/T01_140*.vs')
# all_names[180].append('T0_1/T01_180*.vs')
# all_names[220].append('T0_1/T01_220*.vs')
#    
# 
# sources = [100,140,180]
# 
# dx = 1.0
# 
# depth = np.array([40,48.35,57.7,66.05,75.4,83.75,93.1,101.45,109.8,119.15,127.5,136.85,145.2,154.55,162.9,172.25,180.6,188.95,198.3,206.65,216,224.35,233.7,242.05,250.4])/100
# n_receivers =  np.arange(0,25)+1
# 
# source_depth = [1.0,1.4,1.8] # source depth
# filname = ['Tomo_3D_'+'HW'+str(ss+1)+'_full' for ss in range(0,20)]
# filname.append('Tomo_3D_T0_full')
# 
# 
# n_rec_pole = 25
# 
# iz = 0
# 
# while iz < 21:
# 
#     
# 
#     path = r'C:\Users\obandohe\OneDrive - Stichting Deltares\Documents\DELTARES PROJECTS_2021\005_CROSS_HOLE_ACOUSTICS\ERT_IDAS_EXPERIMENT_PROCESSING\DATABASE_PREPARATION\VS_FILES\01_VS_POLE_03'
#     os.chdir(path)
# 
# 
#     all_sources = np.zeros((75,len(sources)))
#     
#     iik = 0 
#     
#     while iik < len(sources):
#     
#         yt=glob.glob(all_names[sources[iik]][iz])
#         
#         print(yt)
#         
#         all_tt = np.zeros((75,len(yt)))
#         
#         for i in range(0,len(yt)):
#         
#             name_of_vs = yt[i]
#             #print(name_of_vs)
#             #print(i)
#             dist,tt = get_input_data(name_of_vs,dx)
#             all_tt[:,i] = tt
#         
#         corr_tt =  (all_tt - np.min(all_tt,axis=0))+1.0
#         corr_tt_mean = np.mean(corr_tt,axis=1)
#         
#         all_sources[:,iik] = corr_tt_mean
#         
#         iik+=1
#         
#     all_tt = np.reshape(all_sources.T,(75*len(sources),1))
#         
# 
#     xyz = model_wmup5c_3D(depth,source_depth)
#     
#     tt_sel = all_tt
#     
#     n_sources = len(sources)
#     
# 
#     path = r'C:\Users\obandohe\OneDrive - Stichting Deltares\Documents\DELTARES PROJECTS_2021\005_CROSS_HOLE_ACOUSTICS\ERT_IDAS_EXPERIMENT_PROCESSING\INVERSION\INPUT_PYGIMLI\s_100140180'
#     os.chdir(path)
# 
# 
#     create_pgInput_3D_full(xyz,n_sources,tt_sel,filname[iz]+'.csv')
#     
#     iz +=1
# 
# 
# 
# #%%
# 
# 
# plt.plot(corr_tt)
# plt.plot(tt)
# 
# 
# 
# 
# 
# =============================================================================







