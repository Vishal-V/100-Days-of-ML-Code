#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################

clf = SVC(kernel="rbf", C=10000.0)
X = features_train
Y = labels_train
#X = X[:len(X)/100]
#Y = Y[:len(Y)/100]
t = time()
clf.fit(X,Y)
print "Time to train: ", round((time() - t), 3), "s"
t = time()
pred = clf.predict(features_test)
print "Time to predict: ", round((time() - t), 3), "s"
acc = accuracy_score(pred, labels_test)
print(acc)

#########################################################
'''
    no. of Chris training emails:7936
    no. of Sara training emails:7884
    Time to train:  160.197 s
    Time to predict:  16.249 s
    0.9908987485779295

'''

