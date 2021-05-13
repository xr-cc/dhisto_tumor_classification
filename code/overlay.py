#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:32:30 2017

Overlay Images with Manually-Specified Points as Guide

@author: xiranliu
"""
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from os.path import normpath as fn # Fixes window/linux path conventions
import functions as fnc
from scipy.io import loadmat, savemat
from scipy import ndimage


def onclick(event):
    
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print ('x = %d, y = %d'%(ix, iy))

    global coors,spts,dpts,H,spliced
#    x, y = int(ix), int(iy)
    coors.append((int(ix), int(iy)))

    if len(coors) == 8:
        # optimize overlaying based on manually-specified points
        fig.canvas.mpl_disconnect(cid)
        
        print (sample)
#        plt.close(1)
        spts = np.array(coors[0::2])
        dpts = np.array(coors[1::2])
        H,spliced,ratio,corrl,trans_label = fnc.optimizeSplice(img_gray,dh_img,spts,dpts,label,'corrl')
#        print (H)
        
        # overlay display
        fig2 = plt.figure(2)
        plt.subplot(111)
        plt.imshow(dh_img,cmap='Reds',alpha=0.8)
        plt.imshow(spliced,cmap='Blues',alpha=0.4)
        plt.title((sample+' Coregistration (Overlay Display)'))
        plt.show(2)
        fig2.savefig(savepath+sample+'_man_overlay.png')
        
        # checkboard display
        cb1,cb2 = np.meshgrid(np.arange(w),np.arange(h))
        s = 2*int(min(h,w)/3)
        cb = np.logical_xor(cb1%s<s/2,cb2%s<s/2)
        fig3 = plt.figure(3)
        plt.subplot(111)
        plt.imshow((dh_img-0.5)*cb+(1.5-spliced)*np.logical_not(cb),cmap='RdBu')
        plt.title((sample+' Coregistration (Checkboard Display)'))
        plt.show(3)
        fig3.savefig(savepath+sample+'_man_checkboard.png')
        
        # labelling
        trans_label = trans_label.astype('int')
        trans_label = trans_label*(dh_img>0)
        fig3 = plt.figure(4)
        plt.subplot(111)
        plt.imshow(trans_label,cmap='jet',alpha=0.5)
        plt.imshow(dh_img,cmap='Greys',alpha=0.5)
        plt.clim(0,3)
        plt.show(4)
        plt.title(sample)
        fig3.savefig(savepath+sample+'_man_label.png')
        
        # saving
        features = list()
        for m in maps:
            file = sample+'_'+m+'_map_data.mat'
            a = loadmat(datapath+sample+'/'+file)
            features.append(np.array(a['layer']).T)
            
#        plt.close('all')
        features = np.array(features)
        savemat('../samples/'+sample+'_man.mat',{'label':trans_label,'feature':features})
    
    return 

sample = 'B127_1'

samples = ['B92','B94','B95_1','B95_2','B96','B97','B120_1','B120_2','B121','B122_1','B122_2','B122_3','B122_4','B122_4_5','B123','B124_5','B124_7','B125','B126','B127_1','B127_2','B128']

datapath = '../data/'

maps = ['dti_axial', 'fiber_ratio', 'dti_fa', 'restricted_ratio_2', 'fiber1_axial', 'hindered_ratio', 'fiber1_fa', 'water_ratio', 'dti_adc', 'b0', 'restricted_ratio_1', 'fiber1_radial', 'dti_radial']

## H&E IMAGE
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

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show(1)

savepath = '../img/'
fig.savefig(savepath+sample+'_man_pick.png')



