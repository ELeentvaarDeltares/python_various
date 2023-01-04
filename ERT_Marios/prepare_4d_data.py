
import sys
#sys.path.insert(0, r'D:\karaouli\Desktop\Projects\python_tools')


#from rgbcmyk import multidim_intersect_pandas_4d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

import glob
#from pseudo_depth import pseudo_depth
from read_mpt_data import read_mpt_data
#from make_a2d import make_a2d
#from make_loke import make_loke

from make_pyg import make_pyg
from geofac2 import geofac2
# from make_loke_3d import make_loke_3d_bor
# from make_loke import make_loke
# from make_loke_time_lapse import make_loke_time_lapse
# from make_loke_3d import make_loke_3d_time_bor
import pandas as pd
# from make_loke_3d import make_loke_3d_bor_custom
# from make_loke_3d import make_loke_3d_time_bor_custom
# from make_a3d import make_a3d
from make_pyg_3d import make_pyg_3d
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
import numpy as np
import matplotlib.pyplot as plt


def drop_elec(data,elec):
    
    for k in range(0,4):
        i1=np.where(data[:,k]==elec)[0]
        if k==0:
            d1=i1
        else:
            d1=np.r_[d1,i1]
    d1=np.unique(d1)
    # data=np.delete(data, d1,axis=0)
    
    return d1


def sel_elec(data,first,last):
   
    i1=np.where( (data[:,0]>=first) & (data[:,0]<=last)&
                (data[:,1]>=first) & (data[:,1]<=last)&
                (data[:,2]>=first) & (data[:,2]<=last)&
                (data[:,3]>=first) & (data[:,3]<=last))[0]
                
    return i1




#yt=glob.glob(r".\raw\sort\*.data")
yt = glob.glob(r"P:\11208018-016-urban-geophysics\4. Data\ERT\T3-Warm_left\*.data")

# # # yt=yt[0:244]
# yt=yt[-1]
for i in range(0,len(yt)):
    
    data,elec=read_mpt_data(yt[i])

    if i==0:
        out=data[:,5]
    else:
        out=np.c_[out,data[:,5]]


# out=np.reshape(out,(len(out),1))
# a3
# asdadas


new_elec=elec[:,1:]


#fix two missing elec

new_elec=np.r_[
    elec[:31,:],
    np.c_[32,7.75,0,0],
    elec[31:62,:],
    np.c_[64,7.75,1,0],
    elec[62:,:]]

new_elec=new_elec[:,1:]


# out=np.load('out.npy')        
# data=np.load('data.npy')     


coor=np.c_[new_elec[data[:,0].astype('int32')-1],
           new_elec[data[:,1].astype('int32')-1],
           new_elec[data[:,2].astype('int32')-1],
           new_elec[data[:,3].astype('int32')-1]]

gf,gf2=geofac2(coor)

for i in range(0,out.shape[1]):
    out[:,i]=out[:,i]*gf




# gg=np.load('ix_iter_1.npy')


# abmn_9101=np.genfromtxt('clean000.pyg',skip_header=196,skip_footer=1)


# make common
# com,ia,ib=multidim_intersect_pandas_4d(abmn_9101[:,:4], data[:,:4])
# out=out[ib,:]
# data=data[ib,:]

abmn=np.c_[data[:,:4],out] 



# std=np.std(out,axis=1)
# i1=drop_elec(abmn, 166)
# i2=drop_elec(abmn, 81)
# iall=np.unique(np.r_[i1,i2,gg])


# abmn=np.delete(abmn,iall,axis=0)
# out=np.delete(out,iall,axis=0)
# data=np.delete(data,iall,axis=0)

# ii=[]
# for i in range(4,abmn.shape[1]):
#     ix=np.where(abmn[:,i]<0)[0]
#     if len(ix)>1:
#         if i==4:
#             ii=ix
#         else:
#             ii=np.r_[ii,ix]
# ii=np.unique(ii)            
# # sk=abmn[ii,:]





# abmn=np.delete(abmn,ii,axis=0)
# out=np.delete(out,ii,axis=0)
# data=np.delete(data,ii,axis=0)





# for i in range(0,out.shape[0]):
#     plt.plot(out[i,:])
#     plt.title('%d - %d - %d - %d'%(abmn[i,0],abmn[i,1],abmn[i,2],abmn[i,3])) 
#     plt.ylim([0,40])
#     plt.pause(0.1)

#     plt.clf()

#play with in hole

d1=sel_elec(data,1,32)
d2=sel_elec(data,33,64)
d3=sel_elec(data,65,96)
d4=sel_elec(data,97,128)



test1=np.c_[abmn[d1,:]]
test2=np.c_[abmn[d2,:]]
test3=np.c_[abmn[d3,:]]
test4=np.c_[abmn[d4,:]]


s1=np.mean(test1[:,4:],axis=1)
s2=np.mean(test2[:,4:],axis=1)
s3=np.mean(test3[:,4:],axis=1)
s4=np.mean(test4[:,4:],axis=1)

test1c=np.zeros(test1.shape)
test2c=np.zeros(test2.shape)
test3c=np.zeros(test3.shape)
test4c=np.zeros(test4.shape)

test1c[:,:4]=test1[:,:4]
test2c[:,:4]=test2[:,:4]
test3c[:,:4]=test3[:,:4]
test4c[:,:4]=test4[:,:4]



# abmn2=np.copy(abmn)
# for i in range(5,abmn.shape[1]):
#     abmn2[:,i]=abmn2[:,i]/abmn2[:,4]
# abmn2[:,4]=1


# plt.imshow(abmn2[:,4:],vmin=0.8,vmax=1.2,cmap='rainbow')

# for i in range(0,abmn.shape[0]):
#     plt.plot(abmn[i,4:])
#     plt.pause(0.1)
#     plt.clf()



# for i in range(0,test1.shape[0]):
#     ix=np.where((test1[i,4:]>1.2*s1[i])  | (test1[i,4:]<0.8*s1[i])  )[0]
#     if len(ix)>0:
#         test1c[i,ix+4]=1000
#     ix=np.where((test2[i,4:]>1.2*s2[i])  | (test2[i,4:]<0.8*s2[i])  )[0]
#     if len(ix)>0:
#         test2c[i,ix+4]=1000    
#     ix=np.where((test3[i,4:]>1.2*s3[i])  | (test3[i,4:]<0.8*s3[i])  )[0]
#     if len(ix)>0:
#         test3c[i,ix+4]=1000    
#     ix=np.where((test4[i,4:]>1.2*s4[i])  | (test4[i,4:]<0.8*s4[i])  )[0]
#     if len(ix)>0:
#         test4c[i,ix+4]=1000    


# gg=np.load('ix_iter_1.npy')


# for i in range(0,out,shape[0]):
    


# ix=np.where((test1[:,1]-test1[:,0]==5) & (test1[:,2]-test1[:,0]==2) )[0]
# skata=test1[ix,:]
# plt.plot(test1[ix,4:])
# import pygimli as pg
# import pygimli.meshtools as mt
# from pygimli.physics import ert
# data = ert.load("D:/karaouli/Desktop/Projects/salt_water_intrusion/lab/lin1_0000.pyg")
# aaa=np.array(data['valid'])

# # data.removeSensorIdx(6)
# # data.remove((data["a"]==5)*(data["b"]==5)*(data["m"]==5)*(data["n"]==5))
# data.remove(data["a"]==5)
# data.remove(data["b"]==5)
# data.remove(data["m"]==5)
# data.remove(data["n"]==5)


# data.remove(data["a"]==11)
# data.remove(data["b"]==11)
# data.remove(data["m"]==11)
# data.remove(data["n"]==11)

# bbb=np.array(data['valid'])


# test11=np.asarray(np.c_[data['a'],data['b'],data['m'],data['n'],data['rhoa']])
# pg.viewer.mpl.showDataContainerAsMatrix(data, "a", "m", "rhoa",cmap='rainbow')





# ix=np.where((test2[:,1]-test2[:,0]==14) & (test2[:,2]-test2[:,0]==6) )[0]
# skata=test2[ix,:]
# plt.plot(test2[ix,4:])


# ix=np.where((test3[:,1]-test3[:,0]==14) & (test3[:,2]-test3[:,0]==6) )[0]
# skata=test3[ix,:]
# plt.plot(test3[ix,4:])


# ix1=np.where((test4[:,1]-test4[:,0]==14) & (test4[:,2]-test4[:,0]==6) )[0]
# skata1=test4[ix1,:]
# # plt.plot(test4[ix,4:])

# ix1=np.where((test1[:,1]-test1[:,0]==14) & (test1[:,2]-test1[:,0]==6) )[0]
# skata1=test1[ix1,:]


# ix3=np.where((test3[:,1]-test3[:,0]==14) & (test3[:,2]-test3[:,0]==6) )[0]
# skata3=test3[ix3,:]


# ix2=np.where((test2[:,1]-test2[:,0]==14) & (test2[:,2]-test2[:,0]==6) )[0]
# skata2=test2[ix2,:]


# ix4=np.where((test4[:,1]-test4[:,0]==14) & (test4[:,2]-test4[:,0]==6) )[0]
# skata4=test4[ix4,:]



# ix1=np.where((test1[:,1]-test1[:,0]==7) & (test1[:,2]-test1[:,0]==3) )[0]
# skata1=test1[ix1,:]


# ix3=np.where((test3[:,1]-test3[:,0]==7) & (test3[:,2]-test3[:,0]==3) )[0]
# skata3=test3[ix3,:]


# ix2=np.where((test2[:,1]-test2[:,0]==7) & (test2[:,2]-test2[:,0]==3) )[0]
# skata2=test2[ix2,:]


# ix4=np.where((test4[:,1]-test4[:,0]==7) & (test4[:,2]-test4[:,0]==3) )[0]
# skata4=test4[ix4,:]







# for i in range(4,skata2.shape[1]):

#     # plt.clf()

#     plt.subplot(2,2,1)
#     z=0.5*(skata1[:,3]+skata1[:,2])

#     plt.plot(skata1[:,i])
#     plt.ylabel('Resistivity Ohm,')
#     plt.xlabel('Depth')
#     # plt.title(yt[i-4])





#     plt.subplot(2,2,2)
#     z=0.5*(skata2[:,3]+skata2[:,2])-48

#     plt.plot(skata2[:,i])
#     plt.ylabel('Resistivity Ohm,')
#     plt.xlabel('Depth')
#     # plt.title(yt[i-4])



#     plt.subplot(2,2,3)
#     z=0.5*(skata3[:,3]+skata3[:,2])-96

#     plt.plot(skata3[:,i])
#     plt.ylabel('Resistivity Ohm,')
#     plt.xlabel('Depth')
#     # plt.title(yt[i-4])
    
    
#     plt.subplot(2,2,4)
#     z=0.5*(skata4[:,3]+skata4[:,2])-144

#     plt.plot(skata4[:,i])
#     plt.ylabel('Resistivity Ohm,')
#     plt.xlabel('Depth')
    # plt.title(yt[i-4])
    
    
    
    
    # plt.pause(0.2)
    # plt.clf()
    
    
#     plt.plot(test2[ix,:])
    
    
    
#     plt.ylim([0,50])
#     plt.pause(0.1)
#     plt.clf()


def moving_average(x, w):
    return np.convolve(x, np.ones(w), mode='same') / w



# abmn2=np.copy(abmn[:,4:])
# abmn3=np.copy(abmn[:,4:])
# for i in range(0,abmn3.shape[0]):
#     abmn3[i,:]=moving_average(abmn2[i,:], 5)


# test1=abmn2/abmn3
# plt.imshow(test1[:,3:-3],cmap='rainbow',vmin=0.7,vmax=1.3), plt.colorbar()
# # hjtyhu

# i1=np.count_nonzero( (test1<1.2) & (test1>0.8),axis=1)

# ii1=np.where(i1<1100)[0]
# data=np.delete(data,ii1,axis=0)
# out=np.delete(out,ii1,axis=0)

# data = ert.load("test100.pyg")
# a=data['a']
# # data.removeSensorIdx([166])
# # data.removeSensorIdx([81])
# # data.removeSensorIdx([172])

# pg.viewer.mpl.showDataContainerAsMatrix(data, "a", "m", "rhoa",cmap='rainbow')

# filter





# ix=np.where(out[:,0]<0.0)[0]
# ix=np.where(out[:,400]<0.0)[0]


# ix=np.where(out[:,0]<0.3)[0]
# del1=data[ix,:]

# ix=np.where(out[:,0]>100)[0]
# del2=data[ix,:]






# out2=np.copy(out)
# bas=np.copy(out2[:,1])
# for i in range(0,out2.shape[1]):
#     out2[:,i]=out2[:,i]/bas



# fix zeros
# stat=np.zeros((out.shape[0]))
# for i in range(0,out.shape[0]):
#     ix=np.where(out[i,:]>0)[0]
#     x=np.arange(0,out.shape[1])
#     if len(ix>10):
#         test=np.interp(x, x[ix], out[i,ix])
#         # plt.plot(x,out[i,:])
#         # plt.plot(x,test)
#         ix=np.where(out[i,:]<0)[0]
#         out[i,ix]=test[ix]
#         stat[i]=len(ix)
    

# ix=np.where(stat>10)[0]
# out=np.delete(out,ix,axis=0)
# data=np.delete(data,ix,axis=0)
# coor=np.delete(coor,ix,axis=0)

# gg=np.load('ix_iter_2.npy')


# abmn=np.delete(abmn,gg,axis=0)
# out=np.delete(out,gg,axis=0)
# data=np.delete(data,gg,axis=0)


# make_pyg_3d('test101.pyg', new_elec_copy, np.c_[data[:,:4],out[:,100]])
#965 DONE

# iiii=np.where(out[d2,0]>10)[0]

# tt=np.c_[data[d2[iiii],:4],out[d2[iiii]]]
# tt[:,0]=tt[:,0]-32 
# tt[:,1]=tt[:,1]-32 
# tt[:,2]=tt[:,2]-32 
# tt[:,3]=tt[:,3]-32 
'''
for i in range(0,out.shape[1]):
   
    iiii=np.where(out[d1,i]>-np.inf)[0]
    tt=np.c_[data[d1[iiii],:4],out[d1[iiii],i]]
    make_pyg_3d('E:\\Foil2022\\Foil detection 5 cm\\lin1_%04d.pyg'%i, new_elec[:32,:], tt)
    
    iiii=np.where(out[d2,0]>-np.inf)[0]
    tt=np.c_[data[d2[iiii],:4],out[d2[iiii],i]]
    tt[:,0]=tt[:,0]-32 
    tt[:,1]=tt[:,1]-32 
    tt[:,2]=tt[:,2]-32 
    tt[:,3]=tt[:,3]-32    
   
    make_pyg_3d('E:\\Foil2022\\Foil detection 5 cm\\lin2_%04d.pyg'%i, new_elec[:32,:], tt)
    
    
    iiii=np.where(out[d3,0]>-np.inf)[0]
    tt=np.c_[data[d3[iiii],:4],out[d3[iiii],i]]
    tt[:,0]=tt[:,0]-64 
    tt[:,1]=tt[:,1]-64 
    tt[:,2]=tt[:,2]-64 
    tt[:,3]=tt[:,3]-64   
    make_pyg_3d('E:\\Foil2022\\Foil detection 5 cm\\lin3_%04d.pyg'%i, new_elec[:32,:], tt)




    iiii=np.where(out[d4,0]>-np.inf)[0]
    tt=np.c_[data[d4[iiii],:4],out[d4[iiii],i]]
    tt[:,0]=tt[:,0]-96 
    tt[:,1]=tt[:,1]-96 
    tt[:,2]=tt[:,2]-96 
    tt[:,3]=tt[:,3]-96  
    make_pyg_3d('E:\\Foil2022\\Foil detection 5 cm\\lin4_%04d.pyg'%i, new_elec[:32,:], tt)
'''
    