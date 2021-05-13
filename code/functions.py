#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 17:18:31 2017

Functions

@author: xiranliu
"""
import numpy as np
from scipy.signal import convolve2d as conv2
from numpy.linalg import svd
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
from scipy.signal import correlate, correlate2d

# Return magnitude, theta of gradients of image X
def grads(X):
    df = np.float32([[1,0,-1]])
    sf = np.float32([[1,2,1]])
    
    gx = conv2(X,sf.T,'same','symm')
    gx = conv2(gx,df,'same','symm')
    
    gy = conv2(X,sf,'same','symm')
    gy = conv2(gy,df.T,'same','symm')
    
    H = np.sqrt(gx*gx+gy*gy)
    theta = np.arctan2(gy,gx)
    
    return H,theta


# Return image X with outliers removed

def filterOutlier(X,iter=1,more=1):
    X0 = np.copy(X)
    if iter<1:
        iter = 1
    for i in range(iter):
        X0 = filterOutlierOnce1(X0)
    for i in range(more):
        X0 = filterOutlierOnce2(X0)
    return X0

# Iteratively remove outliers in image X
def filterOutlierOnce1(X):
    if len(X.shape)==3:
        h,w,d = X.shape
        Y = (np.all(X>0,axis=-1)).astype(int)
    else:
        h,w = X.shape
        Y = (X>0).astype(int)
    dsum = np.array([[0,1,0],[1,0,1],[0,1,0]])
    g = conv2(Y,dsum,'same')
    Z = g>2
    if len(X.shape)==3:
        Z = np.dstack([Z]*d)
    return np.multiply(X,Z)

# Iteratively remove outliers in image X
def filterOutlierOnce2(X):
    if len(X.shape)==3:
        h,w,d = X.shape
        Y = (np.all(X>0,axis=-1)).astype(int)
    else:
        h,w = X.shape
        Y = (X>0).astype(int)
    dsum = np.array([[1,1,1],[1,0,1],[1,1,1]])
    g = conv2(Y,dsum,'same')
    Z = g>2
    if len(X.shape)==3:
        Z = np.dstack([Z]*d)
    return np.multiply(X,Z)


# Calculate homography that maps from (x,y) to (x',y')
    
def getH(pts):

    x=pts[:,0].reshape((-1,1))
    y=pts[:,1].reshape((-1,1))
    xx=pts[:,2].reshape((-1,1))
    yy=pts[:,3].reshape((-1,1))
    
    z = np.zeros(yy.shape,dtype=np.float32)
    o = np.ones(yy.shape,dtype=np.float32)
    
    r1 = [z,z,z,-x,-y,-o,yy*x,yy*y,yy]
    r2 = [x,y,o,z,z,z,-xx*x,xx*y,-xx]
    r3 = [-yy*x,-yy*y,-yy,xx*x,xx*y,xx,z,z,z]
    
    A = np.concatenate([np.concatenate(r1,axis=1),
                        np.concatenate(r2,axis=1),
                        np.concatenate(r3,axis=1)],axis=0)
    
    u,s,v = svd(A,full_matrices=True)
    
    H=v[-1,:].reshape((3,3))
    
    return H

# Splices the source image into a quadrilateral in the dest imageand return a spliced color image.
def splice(src,dest,spts,dpts,targetZero=False):
    
    ht = src.shape[0]
    wt = src.shape[1]
    
    H = getH(np.concatenate([dpts,spts],axis=1))
    
    x = np.float32(range(np.min(dpts[:,0]),np.max(dpts[:,0])+1))
    y = np.float32(range(np.min(dpts[:,1]),np.max(dpts[:,1])+1))
    
    x,y = np.meshgrid(np.float32(x),np.float32(y))
    x = np.reshape(x,[-1,1])
    y = np.reshape(y,[-1,1])
    
    xyd = np.concatenate([x,y],axis=1)
    xydH = np.concatenate([xyd,np.ones((x.shape[0],1))],axis=1)
    xysH = np.matmul(H,xydH.T).T
    xys = xysH[:,0:2]/xysH[:,2:3]
    
    cnd = np.logical_and(xys[:,0]>0,xys[:,1]>0)
    cnd = np.logical_and(cnd,xys[:,0]<wt-1)
    cnd = np.logical_and(cnd,xys[:,1]<ht-1)
    
    idx = np.where(cnd)[0]
    xyd = np.int64(xyd[idx,:])
    xys = xys[idx,:]
    
    # Bilinear interpolation
    xysf = np.int64(np.floor(xys))
    xysc = np.int64(np.ceil(xys))
    xalph = xys[:,0:1]-np.floor(xys[:,0:1])
    yalph = xys[:,1:2]-np.floor(xys[:,1:2])
    xalph = xalph.flatten()
    yalph = yalph.flatten()    
    
    xff = np.array(src[xysf[:,1],xysf[:,0]])
    xfc = np.array(src[xysf[:,1],xysc[:,0]])
    xcf = np.array(src[xysc[:,1],xysf[:,0]])
    xcc = np.array(src[xysc[:,1],xysc[:,0]])
    a = 1-xalph
    b = 1-yalph
#    comb = dest.copy()
#    comb[xyd[:,1],xyd[:,0],:] = (1-xalph)*((1-yalph)*xff+yalph*xcf) + xalph*((1-yalph)*xfc+yalph*xcc)
    if targetZero:
        comb = np.zeros(dest.shape)
    else:
        comb = np.ones(dest.shape)
    comb[xyd[:,1],xyd[:,0]] = a*(b*xff+yalph*xcf) + xalph*(b*xfc+yalph*xcc)
    
    # corregistration accuracy by ratio of overlaying region
    dt = dest>0
    sp = filterOutlier((1-comb),iter=4,more=2)>0
    diff = dt!=sp
#    accr = 1-np.sum(diff)/np.sum(dt)
    accr = 1-np.sum(diff)/np.sum(dt)*0.5-np.sum(diff)/np.sum(sp)*0.5
#    same = np.logical_and(dt==sp,dt)
#    same_ratio = np.sum(same)/np.sum(dt) 
    
#    accr = np.sum(census(dt)==census(sp))/np.prod(dest.shape)
    
    corrl = crossCorrelation(dt,sp)
    
    return H,comb,accr, corrl
    


def splice2(src,dest,spts,dpts,targetZero=False):
    
    ht = src.shape[0]
    wt = src.shape[1]
    
    H = getH(np.concatenate([dpts,spts],axis=1))
    print (H)
    h,w = dest.shape
    b = int(min(w,h)/8)
#    dpts2 = np.array([[0,0],[0,h-1],[w-1,h-1],[w-1,0]])
#    print (dpts2)
    dpts2 = np.array([[b,b],[b,h-1-b],[w-1-b,h-1-b],[w-1-b,b]])
    
    x = np.float32(range(np.min(dpts2[:,0]),np.max(dpts2[:,0])+1))
    y = np.float32(range(np.min(dpts2[:,1]),np.max(dpts2[:,1])+1))
    
    x,y = np.meshgrid(x,y)
    x = np.reshape(x,[-1,1])
    y = np.reshape(y,[-1,1])

    
    xyd = np.concatenate([x,y],axis=1)
    xydH = np.concatenate([xyd,np.ones((x.shape[0],1))],axis=1)
    xysH = np.matmul(H,xydH.T).T
    xys = xysH[:,0:2]/xysH[:,2:3]

    
    cnd = np.logical_and(xys[:,0]>0,xys[:,1]>0)
    cnd = np.logical_and(cnd,xys[:,0]<wt-1)
    cnd = np.logical_and(cnd,xys[:,1]<ht-1)
    
    idx = np.where(cnd)[0]
    xyd = np.int64(xyd[idx,:])
    xys = xys[idx,:]

    
    # Bilinear interpolation
    xysf = np.int64(np.floor(xys))
    xysc = np.int64(np.ceil(xys))
    xalph = xys[:,0:1]-np.floor(xys[:,0:1])
    yalph = xys[:,1:2]-np.floor(xys[:,1:2])
    xalph = xalph.flatten()
    yalph = yalph.flatten()   

    
    xff = np.array(src[xysf[:,1],xysf[:,0]])
    xfc = np.array(src[xysf[:,1],xysc[:,0]])
    xcf = np.array(src[xysc[:,1],xysf[:,0]])
    xcc = np.array(src[xysc[:,1],xysc[:,0]])
    a = 1-xalph
    b = 1-yalph


    if targetZero:
        comb = np.zeros(dest.shape)
    else:
        comb = np.ones(dest.shape)
    
    return H,1-comb
    

def splice3(src,dest,spts,dpts):
    H = getH(np.concatenate([spts,dpts,],axis=1))
    hs,ws = src.shape
    hd,wd = dest.shape
    b = int(min(ws,hs)/8)
    spts2 = np.array([[b,b],[b,hs-1-b],[ws-1-b,hs-1-b],[ws-1-b,b]])

    x = np.float32(range(np.min(spts2[:,0]),np.max(spts2[:,0])+1))
    y = np.float32(range(np.min(spts2[:,1]),np.max(spts2[:,1])+1))
    x,y = np.meshgrid(x,y)
    x = np.reshape(x,[-1,1])
    y = np.reshape(y,[-1,1])
    xys = np.concatenate([x,y],axis=1)
    xysH = np.concatenate([xys,np.ones((x.shape[0],1))],axis=1)
    xydH = np.matmul(H,xysH.T).T
    xyd = xydH[:,0:2]/xydH[:,2:3]
    
    cnd = np.logical_and(xyd[:,0]>0,xyd[:,1]>0)
    cnd = np.logical_and(cnd,xyd[:,0]<wd-1)
    cnd = np.logical_and(cnd,xyd[:,1]<hd-1)
    idx = np.where(cnd)[0]
    xyd = xyd[idx,:]
    xys = xys[idx,:]

    return xyd



max_shift = 100

# Optimize the transformation
def optimizeSplice(src,dest,spts,dpts,label,meas='ratio',shiftMore=False):
    
    neighbors = [[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]]
    shifts = list()
    for nb in neighbors:
        shifts.append([nb,[0,0],[0,0],[0,0]])
        shifts.append([[0,0],nb,[0,0],[0,0]])
        shifts.append([[0,0],[0,0],nb,[0,0]])
        shifts.append([[0,0],[0,0],[0,0],nb])
    
    if shiftMore:
        shifts = list(np.concatenate((shifts,list(np.array(shifts)*2),list(np.array(shifts)*3),list(np.array(shifts)*4),list(np.array(shifts)*5))))


    num_neighbors = len(shifts)
    
    h = dest.shape[0]
    w = dest.shape[1]
    H,spliced,ratio,corrl = splice(src,dest,spts,dpts)
#    print ('initial dpts:',dpts)
    print ('initial accuracy:',ratio,' correlation:',corrl)
    
    for k in range(max_shift):
        improvement = np.zeros(num_neighbors)-1
        
        for i,shift in enumerate(shifts):
            dpts_neighbor = np.array(dpts.copy()+np.array(shift))
            if np.any(dpts_neighbor[:,0]<0) or np.any(dpts_neighbor[:,0]>=w) or np.any(dpts_neighbor[:,1]<0) or np.any(dpts_neighbor[:,1]>=h):
                break
            H_n,spliced_n,ratio_n,corrl_n = splice(src,dest,spts,dpts_neighbor)
            if meas=='ratio':
                improvement[i] = ratio_n-ratio
            else:
                improvement[i] = corrl_n-corrl
                
            
        if np.all(improvement<0):
#            print (improvement)
            break
        else:
            best_i = np.argmax(improvement)
            best_shift = shifts[best_i]
            dpts = dpts+best_shift
            H,spliced,ratio,corrl = splice(src,dest,spts,dpts)
            
    print ('steps:',k)
    print ('final accuracy:',ratio,' correlation:',corrl,'\n')
    
    # Transform Labels
    ht = src.shape[0]
    wt = src.shape[1]
    
    x = np.float32(range(np.min(dpts[:,0]),np.max(dpts[:,0])+1))
    y = np.float32(range(np.min(dpts[:,1]),np.max(dpts[:,1])+1))
    x,y = np.meshgrid(np.float32(x),np.float32(y))
    x = np.reshape(x,[-1,1])
    y = np.reshape(y,[-1,1])
    
    xyd = np.concatenate([x,y],axis=1)
    xydH = np.concatenate([xyd,np.ones((x.shape[0],1))],axis=1)
    xysH = np.matmul(H,xydH.T).T
    xys = xysH[:,0:2]/xysH[:,2:3]
    
    cnd = np.logical_and(xys[:,0]>0,xys[:,1]>0)
    cnd = np.logical_and(cnd,xys[:,0]<wt-1)
    cnd = np.logical_and(cnd,xys[:,1]<ht-1)
    
    idx = np.where(cnd)[0]
    xyd = np.int64(xyd[idx,:])
    xys = xys[idx,:]
    
    # Bilinear interpolation
    xysf = np.int64(np.floor(xys))
    xysc = np.int64(np.ceil(xys))
    xalph = xys[:,0:1]-np.floor(xys[:,0:1])
    yalph = xys[:,1:2]-np.floor(xys[:,1:2])
    xalph = xalph.flatten()
    yalph = yalph.flatten()    
    
    xff = np.array(label[xysf[:,1],xysf[:,0]])
    xfc = np.array(label[xysf[:,1],xysc[:,0]])
    xcf = np.array(label[xysc[:,1],xysf[:,0]])
    xcc = np.array(label[xysc[:,1],xysc[:,0]])
    a = 1-xalph
    b = 1-yalph

    trans_label = np.zeros(dest.shape)
    trans_label[xyd[:,1],xyd[:,0]] = a*(b*xff+yalph*xcf) + xalph*(b*xfc+yalph*xcc)
    
    return H,spliced,ratio,corrl,trans_label
    


# Label the H&E staining image according to the pathologies marked out by clinicians using dotted red/blue/yellow lines.
    
def label(he_img,sample=''):
    
    height,width = he_img.shape[0],he_img.shape[1]
    
    # red
    red = np.logical_and(np.logical_and(he_img[:,:,0]>190/255,he_img[:,:,1]<50/255),he_img[:,:,2]<80/255).astype('int')
    
    # blue
    blue = np.logical_and(np.logical_and(he_img[:,:,2]>160/255,he_img[:,:,1]<180/255),he_img[:,:,0]<100/255).astype('int')
    
    # yellow
    yellow = np.logical_and(he_img[:,:,0]<250/255,np.logical_and(np.logical_and(he_img[:,:,0]>170/255,he_img[:,:,1]>170/255),he_img[:,:,2]<140/255)).astype('int')
    
    
    # Masking
    d = 5
    kernel = np.float32(np.ones((2*d+1,2*d+1)))
    hf = int(np.floor(d/2))
    kernel[d-hf:d+hf,d-hf:d+hf] = 0
    mid_kernel = 1-kernel
    kernel2 = np.float32(np.ones((2*d+1,2*d+1)))
    
    # RED
    # Connect boundary
    holder = np.zeros(red.shape)
    k = conv2(red,kernel,'valid')
    mid_k = conv2(red,mid_kernel,'valid')
    fill = conv2(red,kernel2,'valid')>0
    holder[d:height-d,d:width-d] = np.logical_and(k>0,mid_k==0)*fill
    holder = holder+red
    holder = conv2(holder,np.ones((d,d)),'same')>0
    # Fill circle
    filled = binary_fill_holes(holder)
    filled = filled*(1-holder)
    r_filled = filterOutlier(filled,iter=2,more=2)
    
    
    # BLUE
    # Connect boundary
    holder = np.zeros(blue.shape)
    k = conv2(blue,kernel,'valid')
    mid_k = conv2(blue,mid_kernel,'valid')
    fill = conv2(blue,kernel2,'valid')>0
    holder[d:height-d,d:width-d] = np.logical_and(k>0,mid_k==0)*fill
    holder = holder+blue
    holder = conv2(holder,np.ones((d,d)),'same')>0
    # Fill circle
    filled = binary_fill_holes(holder)
    filled = filled*(1-holder)
    b_filled = filterOutlier(filled,iter=2,more=2)
    
    
    # YELLOW
    # Connect boundary
    holder = np.zeros(yellow.shape)
    k = conv2(yellow,kernel,'valid')
    mid_k = conv2(yellow,mid_kernel,'valid')
    fill = conv2(yellow,kernel2,'valid')>0
    holder[d:height-d,d:width-d] = np.logical_and(k>0,mid_k==0)*fill
    holder = holder+yellow
    holder = conv2(holder,np.ones((d,d)),'same')>0
    # Fill circle
    filled = binary_fill_holes(holder)
    filled = filled*(1-holder)
    y_filled = filterOutlier(filled,iter=2,more=2)
    
    filled = r_filled+2*b_filled+3*y_filled
    
    fig1 = plt.figure(1)
    plt.imshow(filled,cmap='jet')
#    plt.colorbar()
    plt.clim(0,3)
    fig1.savefig('../img/'+sample+'_mask.png')
    plt.close(fig1)
    
    # display both
    fig2 = plt.figure(2)
    plt.imshow(he_img)
    plt.imshow(filled,cmap='jet',alpha=0.5)
#    plt.colorbar()
    plt.clim(0,3)
    fig2.savefig('../img/'+sample+'_label.png')
#    plt.close(fig2)    
    
    return filled


# Compute cross-correlatino between two images of the same size.
def crossCorrelation(img1,img2):
    
    f1 = np.sqrt(np.sum(img1**2))
    f2 = np.sqrt(np.sum(img2**2))
    cc = np.sum(img1*img2)/(f1*f2)
    
    return cc
