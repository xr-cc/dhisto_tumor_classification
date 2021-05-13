#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:54:34 2017

Mulciclass Classification: 1 vs 2 vs 3
Leave-One-Sample-Out

@author: xiranliu
"""

from scipy.io import loadmat, savemat
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import random


samples = ['B92','B94','B95_1','B95_2','B96','B97','B120_1','B120_2','B121','B122_1','B122_2','B122_3','B122_4','B122_4_5','B123','B124_5','B124_7','B125','B126','B127_1','B127_2','B128']

maps = ['dti_axial', 'fiber_ratio', 'dti_fa', 'restricted_ratio_2', 'fiber1_axial', 'hindered_ratio', 'fiber1_fa', 'water_ratio', 'dti_adc', 'b0', 'restricted_ratio_1', 'fiber1_radial', 'dti_radial']

path = '../samples/'
selection = 'man'

results = list()
for test in samples:
    test_samples = [test]
    print ("Test Sample:",test_samples)
    
    labels,features,coors = np.empty([0,1]),np.empty([0,13]),np.empty([0,2])
    test_labels,test_features,test_coors = np.empty([0,1]),np.empty([0,13]),np.empty([0,2])
    
    perf = list()
    
    for sample in samples:
    
        file = sample+'_'+selection+'_reshape.mat'
        sample_data = loadmat(path+file)
        label = sample_data['label']
        feature = sample_data['feature']
        coor = sample_data['coor']
        
        if sample in test_samples:
            test_labels = np.concatenate((test_labels,label),axis=0)
            test_features = np.concatenate((test_features,feature),axis=0)
            test_coors = np.concatenate((test_coors,coor),axis=0)
        else:
            labels = np.concatenate((labels,label),axis=0)
            features = np.concatenate((features,feature),axis=0)
            coors = np.concatenate((coors,coor),axis=0)
    
    indices = np.arange(len(labels))
    test_indices = np.arange(len(test_labels))
    labels = labels.ravel()
    test_labels = test_labels.ravel()
    
    
#    # Data Statistics
#    mn = np.mean(features,axis=0)
#    minv = np.min(features,axis=0)
#    maxv = np.max(features,axis=0)
#    for i in range(13):
#            print("{:>18}: mean {:+.3f}  min {:+.2f}  max {:+.2f} ".format(maps[i],mn[i],minv[i],maxv[i]))
    
    for k in range(5): 
        
        # Train Test Split
        np.random.seed(k)
        np.random.shuffle(indices)
        np.random.shuffle(test_indices)
        
        X_train, y_train = features[indices,:], labels[indices]
        X_test, y_test = test_features[test_indices,:], test_labels[test_indices]
        
        # Training & Test Data
        if k==0:
            print ("Size of Training Set:",len(y_train))
            print ("#Training 1:",len(np.where(y_train==1)[0]))
            print ("#Training 2:",len(np.where(y_train==2)[0]))
            print ("#Training 3:",len(np.where(y_train==3)[0]))
            
            print ("Size of Test Set:",len(y_test))
            print ("#Test 1:",len(np.where(y_test==1)[0]))
            print ("#Test 2:",len(np.where(y_test==2)[0]))
            print ("#Test 3:",len(np.where(y_test==3)[0]))
        
        # Training
        
        # Uncomment the classifier choosed:
#        clf = SVC(C=1, kernel='rbf', gamma=0.01, class_weight='balanced')
#        clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
#        clf = RandomForestClassifier()
        clf = SGDClassifier(loss="hinge", penalty="l2")
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)  
        accr = accuracy_score(y_test,y_pred)
        cm = confusion_matrix(y_test,y_pred)
        accrs = list()
        accrs.append(accr)
        cls_test = np.unique(y_test)
        cls_pred = np.unique(y_pred)
        ind_accrs = np.zeros(3)
        for i in range(len(cls_pred)):
            if np.sum(cm[i,:])>0:
                ind_accrs[int(cls_pred[i]-1)] = cm[i,i]/np.sum(cm[i,:])
            else:
                ind_accrs[int(cls_pred[i]-1)] = 0
        accrs.append(ind_accrs)
        perf.append(accrs)
        
    results.append(perf)

print (clf)
re = results.copy()
i = 0
for s in re:
    print ("Sample",samples[i])
    overall_accr = list()
    individual_accr = list()
    for r in s:
        overall_accr.append(r[0])
        individual_accr.append(r[1])
    overall_accr = np.array(overall_accr)
    individual_accr = np.array(individual_accr)
    overall_accr = np.mean(overall_accr)
    individual_accr = np.mean(individual_accr,axis=0)
    print ("Accuracies:",overall_accr,"|",individual_accr)
    i = i+1