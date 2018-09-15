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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

X = features_train
Y = labels_train
Z = features_test
V = labels_test
print len(X[0]) #The number of features considered

### The number of features determine the complexity of the decision tree
### Thus I have chosen a smaller number 379 instead of 3785 features
########################## DECISION TREE #################################

clf = DecisionTreeClassifier(min_samples_split=40)
t = time()
clf.fit(X,Y)
print "The time to train is: ", round(time() - t,3)
t = time()
pred = clf.predict(Z)
print "The time to train is: ", round(time() - t,3)
accuracy = accuracy_score(pred,V)
print accuracy

"""
    no. of Chris training emails:7936
    no. of Sara training emails:7884
    The time to train is:  64.327
    The time to train is:  0.041
    0.9783845278725825
    
"""