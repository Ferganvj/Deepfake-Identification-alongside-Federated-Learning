# Intelligent Systems and Assistive Technology project for the University of the Western Cape - COS 731 + 732 (Hons project)

## Introduction

We present Deepfake Identification alongside Federated Learning to detect synthetic images. Machine learning plays a significant role in assisting with complicated and convoluted problems that breaches human ability. The precision and consistency with which technology can reliably and easily discern the integrity of digitalized visual media is thus critical.


## Requirements
* ``` Olivetti faces dataset ```
* ``` Python 3.8 (or above) ```

## Dataset specifications
There are 400 sample images where there are 10 different images of 40 distinct subjects. The subjects vary from lighting, facial expressions of open eyes, closed eyes, smiling, not smiling, and facial details such as glasses and/or no glasses.

# Deepfake Identification system setup:
It is good practice to ensure and update all packages frequently.

## Dataset setup:
Please ensure you have correctly input/locate the appropriate directory where the dataset is located. By ensuring this it allows for the data to be loaded and manipluated.

## Using Google Colab:
* ``` Upload to session storage ```
* ``` Create a new folder named: Face_Recognition ```
* ``` Navigate in the directory where the dataset is located ```
* ``` Navigate to "Olivetti Dataset" ```
* ``` Upload "olvetti_faces.npy" and "olivetti_faces_target.npy" ```
* ``` Move the two (2) .npy files into the Face_Recognition folder ```

## Ensure that the dataset is properly loaded by running the following code in the notebook:
```
import os
mydir = r'/content/Face_Recognition'
myfile = 'Face_Recognition'
Face_Recognition_path = os.path.join(mydir, myfile)
```
To load the image data and verify the data run the following code:
```
data = np.load("/content/Face_Recognition/olivetti_faces.npy")
target = np.load("/content/Face_Recognition/olivetti_faces_target.npy")
```
Verify the data by running:
```
print("There are {} images in the dataset".format(len(data)))
print("There are {} unique targets in the dataset".format(len(np.unique(target))))
print("The size of each image is: {}x{}".format(data.shape[1],data.shape[2]))
```
Preceeding the successful loading and verification of the dataset the following will be displayed:
* ``` Number of images in the dataset ```
* ``` Number of unique targets in the dataset ```
* ``` The size of each image in the dataset ```

## Then to simply install other libs:
In terminal to:

Install Numpy for linear algebra
* ``` pip install Numpy ```

Install Pandas for data processing
* ``` pip install Pandas ```

# Machine Learning setup
Please ensure to enable Machine Learning features for testing and training purposes.

Within the code cell import the following machine learning features:
```
from sklearn import metrics
```
Architectural Models:
```
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
```
Train-test split on the dataset
```
from sklearn.model_selection import train_test_split
```
Graph Visualization, PCA for component analysis, Heatmap of Confusion Matrix
```
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
```

Cross Validation of different architectural models
```
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
```

Saving the models as Pickle files (.plk)
```
import pickle
import joblib
```

# Federated Learning environment setup
Import the following libraries for federated learning setup.
```
import numpy as np
import random
import cv2
import os
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
```

## Conclusions
B.Sc (Hons) in Computer Science for University of Western Cape, Cape Town, 7535, South Africa.

Honours project: ISAT project - Deepfake Identification alongside Federated Learning 

Subject code: COS731 (Semester 1) + COS732 (Semester 2)

Student:
* Fergan van Jaarsveld - 4164150@myuwc.ac.za

Supervisors:
* Dr. Olasupo Ajayi - ooajayi@uwc.ac.za
* Hlonipani Maluleke - hhmaluleke@uwc.ac.za

Co-Supervisor:
* Prof. Antoine Bagula - abagula@uwc.ac.za