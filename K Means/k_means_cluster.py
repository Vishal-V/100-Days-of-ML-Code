#!/usr/bin/python 

""" 
    k-means clustering mini-project.
"""

import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )

salaries = []
count = 0
for sample in data_dict.items():
    val = sample[1]['salary']
    if(val != 'NaN'):
        salaries.insert(count, val)
    count += 1
salaries.sort(reverse=True)
### Printing the max and min values for salary
### index 1 because index 0 is total salary of company
print salaries[1]," ", salaries[len(salaries)-1]

### there's an outlier to be removed
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2, feature_3]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:

for f1, f2, _ in finance_features:
    plt.scatter( f1, f2 )
plt.ylabel("exercised_stock_options")
plt.xlabel("salary")
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from time import time
target, feature = targetFeatureSplit(data)
X_train, X_test, Y_train, Y_test = train_test_split(feature, target)
cluster = KMeans(n_clusters=3, max_iter=300, n_init=10)
t = time()
cluster.fit(X_train, Y_train)
print "Time to train: ", round(time() - t, 3)
pred = cluster.predict(X_test)


def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()
    
### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file

try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters_3ft.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"

