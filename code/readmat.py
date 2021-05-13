#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:03:34 2017

Read .mat files of D-Histo data

@author: xiranliu
"""
import scipy
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
import re


## Read Map from Mat File
samples = ['B92','B94','B95_1','B95_2','B96','B97','B120_1','B120_2','B121','B122_1','B122_2','B122_3','B122_4','B122_4_5','B123','B124_5','B124_7','B125','B126','B127_1','B127_2','B128']
datapath = '../data/'

for i,sample in enumerate(samples[0:1]):
    files = listdir(datapath+sample+'/')
#    files = [re.sub(r'_map_data.mat$', '',f) for f in files]
#    features = [re.sub(r'B\d+_', '',f) for f in files]

#    s = sample+'/'+sample+'_restricted_ratio_2_map_data.mat'
##    s = sample+'/'+sample+'_dti_adc_map_data.mat'
##    s = sample+'/'+sample+'_water_ratio_map_data.mat'
##    s = sample+'/'+sample+'_b0_map_data.mat'
#    
    a = scipy.io.loadmat(datapath+s)
    data = np.array(a['layer'])
    data = data.T/np.max(data)
    #plt.imshow(a,cmap='gist_rainbow')
    scipy.io.savemat(datapath+sample+'.mat',{'data':data})
    plt.figure(i)
    plt.imshow(data,cmap='rainbow')
    
    