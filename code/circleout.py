#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:32:30 2017

Generate labeling masking based on pathologies marked out on H&E staining image

@author: xiranliu
"""
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from PIL import Image
from os.path import normpath as fn # Fixes window/linux path conventions
import functions as fnc
from scipy.io import loadmat
from scipy import ndimage
import itertools
from scipy.signal import convolve2d as conv2
from scipy.ndimage.morphology import binary_fill_holes


sample = 'B127_1'

#samples = ['B92','B94','B95_1','B95_2','B96','B97','B120_1','B120_2','B121','B122_1','B122_2','B122_3','B122_4','B122_4_5','B123','B124_5','B124_7','B125','B126','B127_1','B127_2','B128']

#for sample in samples:
print (sample)
# H&E Image
he_img = np.float32(imread(fn('../he/'+sample+'.jpg')))/255.
# Labeling on H&E
fnc.label(he_img,sample)


#img_gray = np.float32(imread(fn('../he/'+sample+'.jpg'),as_grey=True))
#
#
### D-HISTO MAP
#data = loadmat('../data/'+sample+'.mat')
#data = data['data']
#dh_img = fnc.filterOutlier(data,iter=3,more=2) # filter outliers
#h,w = dh_img.shape
#
## Placeholders for Parameters
#coors, spts, dpts = [], [], []
#H,spliced = None,None
#
#
### Display images for manually-specified points
#fig = plt.figure(1)
#
#plt.subplot(121)
#plt.title((sample+' H&E Image'))
#plt.imshow(img_gray,cmap='gray')
#
#plt.subplot(122)
#plt.title((sample+' D-Histo Map'))
#plt.imshow(dh_img,cmap='gray')
#
##cid = fig.canvas.mpl_connect('button_press_event', onclick)
#
#plt.show(1)
#
#fig = plt.figure(2)
#plt.imshow(he_img)
#
#
#height,width = he_img.shape
#
## red
#red = np.logical_and(np.logical_and(he_img[:,:,0]>200/255,he_img[:,:,1]<50/255),he_img[:,:,2]<50/255).astype('int')
#
## blue
#blue = np.logical_and(np.logical_and(he_img[:,:,2]>200/255,he_img[:,:,1]<200/255),he_img[:,:,0]<100/255).astype('int')
#
## yellow
#yellow = np.logical_and(np.logical_and(he_img[:,:,0]>200/255,he_img[:,:,1]>200/255),he_img[:,:,2]<150/255).astype('int')
#
#
## Masking
#d = 5
#kernel = np.float32(np.ones((2*d+1,2*d+1)))
#hf = int(np.floor(d/2))
#kernel[d-hf:d+hf,d-hf:d+hf] = 0
#mid_kernel = 1-kernel
#kernel2 = np.float32(np.ones((2*d+1,2*d+1)))
#
## RED
## Connect boundary
#holder = np.zeros(red.shape)
#k = conv2(red,kernel,'valid')
#mid_k = conv2(red,mid_kernel,'valid')
#fill = conv2(red,kernel2,'valid')>0
#holder[d:height-d,d:width-d] = np.logical_and(k>0,mid_k==0)*fill
#holder = holder+red
#holder = conv2(holder,np.ones((d,d)),'same')>0
## Fill circle
#filled = binary_fill_holes(holder)
#filled = filled*(1-holder)
#r_filled = fnc.filterOutlier(filled,iter=2,more=2)
#
#
## BLUE
## Connect boundary
#holder = np.zeros(blue.shape)
#k = conv2(blue,kernel,'valid')
#mid_k = conv2(blue,mid_kernel,'valid')
#fill = conv2(blue,kernel2,'valid')>0
#holder[d:height-d,d:width-d] = np.logical_and(k>0,mid_k==0)*fill
#holder = holder+blue
#holder = conv2(holder,np.ones((d,d)),'same')>0
## Fill circle
#filled = binary_fill_holes(holder)
#filled = filled*(1-holder)
#b_filled = fnc.filterOutlier(filled,iter=2,more=2)
#
#
## YELLOW
## Connect boundary
#holder = np.zeros(yellow.shape)
#k = conv2(yellow,kernel,'valid')
#mid_k = conv2(yellow,mid_kernel,'valid')
#fill = conv2(yellow,kernel2,'valid')>0
#holder[d:height-d,d:width-d] = np.logical_and(k>0,mid_k==0)*fill
#holder = holder+yellow
#holder = conv2(holder,np.ones((d,d)),'same')>0
## Fill circle
#filled = binary_fill_holes(holder)
#filled = filled*(1-holder)
#y_filled = fnc.filterOutlier(filled,iter=2,more=2)
#
#filled = r_filled+2*b_filled+3*y_filled
#
#fig = plt.figure(3)
#plt.imshow(filled)
#plt.colorbar()
#
#
## display both
#fig = plt.figure(4)
#plt.imshow(he_img)
#plt.imshow(filled,alpha=0.5)
#plt.colorbar()

