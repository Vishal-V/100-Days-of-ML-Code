#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

### your code below
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from time import time
X = []
Y = []
i = 0

### Initial plot to visualize outliers and gather the features and labels
for sample in data:
    salary = sample[0]
    bonus = sample[1]
    if(i == 67):
        i += 1
        continue
    X.insert(i, salary)
    Y.insert(i, bonus)
    i += 1
    plt.scatter(salary,bonus)    
    
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

### Splitting the data and fitting it into the regresssion
target, feature = targetFeatureSplit(data)
X_train, X_test, Y_train, Y_test = train_test_split(feature, target, random_state=42)
reg = LinearRegression()
t = time()
reg.fit(X_train, Y_train)
print "Time to train: ", round(time() - t, 3)
t = time()
pred = reg.predict(X_test)
print "Time to train: ", round(time() - t, 3)
