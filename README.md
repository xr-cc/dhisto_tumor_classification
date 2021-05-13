
This repository contains code used for registration of D-Histo image and classification of brain tumor pathology. Part of the code is used in the analysis in paper [Diffusion Histology Imaging Combining Diffusion Basis Spectrum Imaging (DBSI) and Machine Learning Improves Detection and Classification of Glioblastoma Pathology](https://clincancerres.aacrjournals.org/content/26/20/5388.article-info) by Ye **et al.** .

All files run on Python 3.6.1 using Anaconda. Packages used include scipy, numpy, sklearn.


---Data---

Because of the confidentiality of medical data, original data files are not included. Selected image files are included for demo.


---Directories---

Code is under directory "code". 

Other directories on the same level are: 
"data" which contains original data files, 
"he" which contains the original H&E staining images, 
"img" where the output images will be put 
and "sample" where the labeled data files will be put.


---Instructions---

1. Run circleout.py to generate masking images with labeling information.


2. Run overlayautoDemo.py to register the selected sample.

* "overlayauto.py" or "overlay.py" is what should be run. 
* Because original data files are not provided, these file can not be able to run in demo.)
* The labeled data files are provided directly for the next step.


3. Run dataAnalysisDemo.py to process labeled data and save them as format suitable for leanrning algorithms.

* "dataAnalysis.py" is what should be run. 
* Because original data files are not provided, the file can not be able to run in demo.
* The processed data files are provided directly for the next step.


4. Run multiClsLMSO.py (or multiClsLOSO.py) to see the classification results using "leave-multiple-samples-out" (or "leave-one-sample-out") strategy.



---Files and Usages---

readmat.py
-This file read .mat files of D-Histo data and put them into desired directory.

functions.py
- This file contains functions called by other files in this project.

circle.py
- This file generates labeling masking of samples based on pathologies marked out on their H&E staining image.
- It will save the masking images under directory "img".

overlay.py
- This file registers the specified sample image. 
- It requires a manual selection of initial set of matching points (the four pairs of outermost boundary points on two images).
- It will print out the registration performances.
- It will save the registered images under directory "img" and save the labeled data together with features under directory "samples".

overlayauto.py
- This file registers all sample images automatically. 
- It will automatically selects an initial set of matching points.
- It will print out the registration performances.
- It will save the registered images under directory "img" and save the labeled data together with features under directory "samples".

overlayautoDemo.py
- This file is a simplified version of overlayauto.py for use in demo with selected sample files.

dataAnalysis.py
- This file processes the labeled data files and does simple analysis on the data.
- It prints out statistics for all D-Histo maps provided for each sample.
- It reshapes the data into input format than can be used by learning algorithms (feature:nxd, label:nx1, coor:nx2, where n is number of valid voxels and d is number of D-Histo maps) and save them under directory "samples".

dataAnalysisDemo.py
- This file is a simplified version of dataAnalysis.py for use in demo with selected sample files.
- It also shows the original masking image and the masking reconstructed from the saved coordinates in data.

multiClsLOSO.py
- This file implements the multiclass classification using four classifiers.
- It trains the data and tests using "leave-one-sample-out" strategies.
- It prints out the averaged accuracies.

multiClsLMSO.py
- This file implements the multiclass classification using four classifiers.
- It trains the data and tests using "leave-multiple-samples-out" strategies.
- The combination of test samples are specified by user.
- It prints out the averaged accuracies.

binaryCls.py
- This file implements the pairwise binary classification using four classifiers.
- The two classes that are compared are specified by user.
- It prints out the accuracy score, precision and recall.

*Some other files that are not directly required for the completness of this project is not included. They can be provided upon request.


--------------------------------------
Author: Xiran Liu (liu.xiran@wustl.edu)