#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:54:34 2017

Binary Classification: 1 vs 2, 2 vs 3, 1 vs 3
Test samples Need to be specified.

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

selected_samples = ['B92','B95_1','B95_2','B96','B120_1','B120_2','B121','B122_1','B122_4','B123','B124_5','B124_7','B125','B126','B127_1','B127_2','B128']

class1 = 1
class2 = 2
class3 = 3 # class that is not used

print ("Class",class1,"vs. Class",class2)

labels,features,coors = np.empty([0,1]),np.empty([0,13]),np.empty([0,2])
test_labels,test_features,test_coors = np.empty([0,1]),np.empty([0,13]),np.empty([0,2])


test_samples = ['B96','B122_3','B125']

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

# Remove nonused classes
nonused_idx = np.where(labels!=class3)[0]
labels = labels[nonused_idx]
features = features[nonused_idx,:]
coors = coors[nonused_idx,:]
nonused_test_idx = np.where(test_labels!=class3)[0]
test_labels = test_labels[nonused_test_idx]
test_features = test_features[nonused_test_idx,:]
test_coors = test_coors[nonused_test_idx,:]


indices = np.arange(len(labels))
test_indices = np.arange(len(test_labels))
labels = labels.ravel()
test_labels = test_labels.ravel()


## Data
#mn = np.mean(features,axis=0)
#minv = np.min(features,axis=0)
#maxv = np.max(features,axis=0)
#for i in range(13):
#        print("{:>18}: mean {:+.3f}  min {:+.2f}  max {:+.2f} ".format(maps[i],mn[i],minv[i],maxv[i]))


## Train Test Split
np.random.shuffle(indices)
np.random.shuffle(test_indices)
X_train, y_train = features[indices,:], labels[indices]
X_test, y_test = test_features[test_indices,:], test_labels[test_indices]



# Training & Test Data
print ("Size of Training Set:",len(y_train))
print ("#Training {}:".format(str(class1)),len(np.where(y_train==class1)[0]))
print ("#Training {}:".format(str(class2)),len(np.where(y_train==class2)[0]))

print ("Size of Test Set:",len(y_test))
print ("#Test {}:".format(str(class1)),len(np.where(y_test==class1)[0]))
print ("#Test {}:".format(str(class2)),len(np.where(y_test==class2)[0]))

       
# Training

#clf = SVC(C=1, kernel='rbf', gamma=0.01, class_weight='balanced')
#clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
#clf = RandomForestClassifier()
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)  
accr = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)
ps = precision_score(y_test,y_pred)
rc = recall_score(y_test,y_pred)
print (clf)
print ("Accuracy",accr)
print ("Confusion Matrix")
print (cm)
print ("Precision Score",ps)
print ("Recall Score",rc)