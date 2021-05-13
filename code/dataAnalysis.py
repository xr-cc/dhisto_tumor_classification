#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:48:02 2017

Process labeled data files and do simple analysis.

@author: xiranliu
"""
from scipy.io import loadmat, savemat
import numpy as np
import matplotlib.pyplot as plt

samples = ['B92','B94','B95_1','B95_2','B96','B97','B120_1','B120_2','B121','B122_1','B122_2','B122_3','B122_4','B122_4_5','B123','B124_5','B124_7','B125','B126','B127_1','B127_2','B128']

maps = ['dti_axial', 'fiber_ratio', 'dti_fa', 'restricted_ratio_2', 'fiber1_axial', 'hindered_ratio', 'fiber1_fa', 'water_ratio', 'dti_adc', 'b0', 'restricted_ratio_1', 'fiber1_radial', 'dti_radial']

path = '../samples/'
selection = 'man'


for sample in samples:

    print ('SAMPLE ',sample)
    file = sample+'_'+selection+'.mat'
    sample_data = loadmat(path+file)
    label = sample_data['label']
    feature = sample_data['feature']
    
    d,m,n = feature.shape
    
    # Reshape
    coor2, coor1 = np.meshgrid(np.arange(n),np.arange(m))
    coors = np.concatenate((coor1.reshape(-1,1),coor2.reshape(-1,1)),axis=1)
    label_flat = np.reshape(label,(-1,1))
    feature_flat = np.reshape(feature,(feature.shape[0],-1)).T
    
    # remove noise 
    for i in [1,2,3]:
        idx = np.where(label_flat==i)[0]
        non_idx = np.where(label_flat!=i)[0]
        if len(idx)>0 and len(idx)<20:
            label_flat = label_flat[non_idx]
            feature_flat = feature_flat[non_idx,:]
            coors = coors[non_idx,:]
    
    # Valid
    
    valid_indices = np.where(label_flat>0)[0]
    idx1 = np.where(label_flat==1)[0]
    idx2 = np.where(label_flat==2)[0]
    idx3 = np.where(label_flat==3)[0]
    
    
    print ("Valid voxels: ",len(valid_indices))
    print ("Class 1: ",len(idx1))
    print ("Class 2: ",len(idx2))
    print ("Class 3: ",len(idx3))
    
    
    # Value
    mn = np.mean(feature_flat,axis=0)
    minv = np.min(feature_flat,axis=0)
    maxv = np.max(feature_flat,axis=0)
    for i in range(d):
        print("{:>18}: mean {:+.3f}  min {:+.2f}  max {:+.2f} ".format(maps[i],mn[i],minv[i],maxv[i]))
        
    # Filter
    valid_label = label_flat[valid_indices]
    valid_feature = feature_flat[valid_indices,:]
    valid_coor = coors[valid_indices,:]
    # value
    print ('[valid]')
    valid_mn = np.mean(valid_feature,axis=0)
    valid_minv = np.min(valid_feature,axis=0)
    valid_maxv = np.max(valid_feature,axis=0)
    for i in range(d):
        print("{:>18}: mean {:+.3f}  min {:+.2f}  max {:+.2f} ".format(maps[i],valid_mn[i],valid_minv[i],valid_maxv[i]))
    
    reshape_file = sample+'_'+selection+'_reshape.mat'
    savemat(path+reshape_file,{'label':valid_label,'feature':valid_feature,'coor':valid_coor})
    
    ## Reconstruct Masking TEST
    #masking = np.zeros((m,n))
    #for c in range(len(valid_coor)):
    #    masking[valid_coor[c,0],valid_coor[c,1]] = valid_label[c]
    #plt.figure(1)
    #plt.imshow(masking)
    #plt.figure(2)
    #plt.imshow(label)