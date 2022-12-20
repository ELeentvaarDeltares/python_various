# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 11:33:57 2018

@author: karaouli
"""
import numpy as np




def geofac2(coords):
    A=coords[:,[0,1,2]]
    B=coords[:,[3,4,5]]
    M=coords[:,[6,7,8]]
    N=coords[:,[9,10,11]]
    
    
    AM=dis_point(A,M)
    AN=dis_point(A,N)
    BM=dis_point(B,M)
    BN=dis_point(B,N)
    
#    if AM>0:
#        AM=1/AM
#    if AN>0:
#        AN=1/AN
#    if BM>0:
#        BM=1/BM
#    if BN>0:
#        BN=1/BN
    
    AM[AM>0]=1/AM[AM>0]
    AN[AN>0]=1/AN[AN>0]
    BM[BM>0]=1/BM[BM>0]
    BN[BN>0]=1/BN[BN>0]
    
    
    #gf=1/dis_point(A,M) - 1/dis_point(A,N) - 1/dis_point(B,M) + 1/dis_point(B,N)
    gf=AM - AN - BM + BN
    
    
    
    # add reflection from top
    A[:,2]=-A[:,2];
    B[:,2]=-B[:,2];
    #M[:,2]=-M[:,2];
    #N[:,2]=-N[:,2];
    
    AM=dis_point(A,M)
    AN=dis_point(A,N)
    BM=dis_point(B,M)
    BN=dis_point(B,N)
    
    
#    if AM>0:
#        AM=1/AM
#    if AN>0:
#        AN=1/AN
#    if BM>0:
#        BM=1/BM
#    if BN>0:
#        BN=1/BN
    AM[AM>0]=1/AM[AM>0]
    AN[AN>0]=1/AN[AN>0]
    BM[BM>0]=1/BM[BM>0]
    BN[BN>0]=1/BN[BN>0]
    
    
    
    
    #gf2=gf+ 1/dis_point(A,M) - 1/dis_point(A,N) - 1/dis_point(B,M) + 1/dis_point(B,N)
    gf2=gf+ AM - AN - BM + BN
    
    gf=2*np.pi/gf
    gf2=4*np.pi/gf2
    
    return gf,gf2
    
    
    
    
    
    
def dis_point(p1,p2):
    dis=np.sqrt(np.power(p1[:,0]-p2[:,0],2) + np.power(p1[:,1]-p2[:,1],2) + np.power(p1[:,2]-p2[:,2],2) )
    return dis
    
    