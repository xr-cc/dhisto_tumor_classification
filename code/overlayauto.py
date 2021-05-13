#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:32:30 2017

Overlay Images with Automatically-Selected Points as Guide

@author: xiranliu
"""
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from os.path import normpath as fn # Fixes window/linux path conventions
import functions as fnc
from scipy.io import loadmat, savemat
from scipy import ndimage



samples = ['B92','B94','B95_1','B95_2','B96','B97','B120_1','B120_2','B121','B122_1','B122_2','B122_3','B122_4','B122_4_5','B123','B124_5','B124_7','B125','B126','B127_1','B127_2','B128']

savepath = '../img/'
datapath = '../data/'

maps = ['dti_axial', 'fiber_ratio', 'dti_fa', 'restricted_ratio_2', 'fiber1_axial', 'hindered_ratio', 'fiber1_fa', 'water_ratio', 'dti_adc', 'b0', 'restricted_ratio_1', 'fiber1_radial', 'dti_radial']


for sample in samples[:]:
    print (sample)

    ## H&E IMAGE
    #he_img = np.float32(imread(fn('../he/'+sample+'.jpg')))/255.
    img_gray = np.float32(imread(fn('../he/'+sample+'.jpg'),as_grey=True))
    
    he_color_img = np.float32(imread(fn('../he/'+sample+'.jpg')))/255
    label = fnc.label(he_color_img,sample)
    
    
    ## D-HISTO MAP
    data = loadmat('../data/'+sample+'.mat')
    data = data['data']
    dh_img = fnc.filterOutlier(data,iter=3,more=2) # filter outliers
    h,w = dh_img.shape
    
    # Placeholders for Parameters
    coors, spts, dpts = [], [], []
    H,spliced = None,None
    
    
    ## Display images for manually-specified points
    fig = plt.figure(1)
    
    plt.subplot(121)
    plt.title((sample+' H&E Image'))
    plt.imshow(img_gray,cmap='gray')
    
    plt.subplot(122)
    plt.title((sample+' D-Histo Map'))
    plt.imshow(dh_img,cmap='gray')
    
    #cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    plt.show(1)
    fig.savefig(savepath+sample+'_auto_comp.png')
    
    ## Clean H&E image
    he_img = 1-img_gray
    hs,ws = he_img.shape
    edge = 10
    rm_edge = np.ones(he_img.shape)
    rm_edge[:,:edge] = 0
    rm_edge[:,-edge:] = 0
    rm_edge[:edge,:] = 0
    rm_edge[-edge:,:] = 0
    he_img_filtered = fnc.filterOutlier(he_img,iter=4,more=2)*rm_edge
    fig = plt.figure(2)
    plt.imshow(he_img_filtered)
    
    
    #he
    a = int(min(ws,hs)/8)
    spts = np.array([[a,a],[a,hs-1-a],[ws-1-a,hs-1-a],[ws-1-a,a]])
    #dh
    b = int(min(w,h)/8)
    dpts = np.array([[b,b],[b,h-1-b],[w-1-b,h-1-b],[w-1-b,b]])
    
    
    ## Co-rgistration and Save
    
    # optimize overlaying based on automatically-selected points
    H,spliced,ratio,corrl,trans_label = fnc.optimizeSplice(img_gray,dh_img,spts,dpts,label,'ratio',shiftMore=True)
#    print (H)
    
    # overlay display
    fig2 = plt.figure(2)
    plt.subplot(111)
    plt.imshow(dh_img,cmap='Reds',alpha=0.8)
    plt.imshow(spliced,cmap='Blues',alpha=0.6)
    plt.show(2)
    plt.title(sample)
    fig2.savefig(savepath+sample+'_auto_overlay.png')
    
    # checkboard display
    cb1,cb2 = np.meshgrid(np.arange(w),np.arange(h))
    s = 2*int(min(h,w)/3)
    cb = np.logical_xor(cb1%s<s/2,cb2%s<s/2)
    fig3 = plt.figure(3)
    plt.subplot(111)
    plt.imshow((dh_img-0.5)*cb+(1.5-spliced)*np.logical_not(cb),cmap='RdBu')
    plt.show(3)
    plt.title(sample)
    fig3.savefig(savepath+sample+'_auto_checkboard.png')
    
    # labelling
    trans_label = trans_label.astype('int')
    trans_label = trans_label*(dh_img>0)
    fig3 = plt.figure(4)
    plt.subplot(111)
    plt.imshow(trans_label,cmap='jet',alpha=0.5)
    plt.imshow(dh_img,cmap='Greys',alpha=0.5)
    plt.clim(0,5)
    plt.show(4)
    plt.title(sample)
    fig3.savefig(savepath+sample+'_auto_label.png')
    
    ##Save
    features = list()
    for m in maps:
        file = sample+'_'+m+'_map_data.mat'
        a = loadmat(datapath+sample+'/'+file)
        features.append(np.array(a['layer']).T)
        
    plt.close('all')
    features = np.array(features)
    savemat('../samples/'+sample+'_auto.mat',{'label':trans_label,'feature':features})
    


