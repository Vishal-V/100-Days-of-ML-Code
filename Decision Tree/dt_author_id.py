#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

X = features_train
Y = labels_train
Z = features_test
V = labels_test

########################## DECISION TREE #################################

### your code goes here--now create 2 decision tree classifiers,
### one with min_samples_split=2 and one with min_samples_split=50
### compute the accuracies on the testing data and store
### the accuracy numbers to acc_min_samples_split_2 and
### acc_min_samples_split_50, respectively
tree1 = tree.DecisionTreeClassifier(min_samples_split=2)
tree2 = tree.DecisionTreeClassifier(min_samples_split=50)
tree1.fit(X,Y)
tree2.fit(X,Y)
pred1 = tree1.predict(Z)
pred2 = tree2.predict(Z)
acc_min_samples_split_2 = accuracy_score(pred1, V)
acc_min_samples_split_50 = accuracy_score(pred2, V)



