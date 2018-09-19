#!/usr/bin/python

"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

# Time to split the data!
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, random_state=42)

# Creating the classifier, fitting the data and predicting from the test set
from sklearn.tree import DecisionTreeClassifier
from time import time
clf = DecisionTreeClassifier(min_samples_split=40)
t = time()
clf.fit(X_train, Y_train)
print "Time to train: ", round(time()-t, 3)
t = time()
pred = clf.predict(X_test)
print "Time to train: ", round(time()-t, 3)

# Print the accuracy score for the decision tree
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, Y_test)
print acc