# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 19:01:05 2018

Tensorflow implementation of the iris dataset classification

@author: Vishal
"""

#Using a linear classifier
import tensorflow.contrib.learn as tf
from sklearn import datasets, metrics
iris = datasets.load_iris()
clf = tf.TensorFlowLinearClassifier(n_classes=3)
clf.fit(iris.data, iris.target)
acc = metrics.accuracy_score(iris.target, clf.predict(iris.data))
print(f'{acc}')

#Using a linear regressor
import tensorflow.contrib.learn as tf
from sklearn import datasets, metrics, preprocessing, cross_validation
iris = datasets.load_iris()
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(iris.data)
labels = iris.target
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)
clf = tf.TensorFlowLinearRegressor()
clf.fit(features_train, labels_train)
accuracy = metrics.accuracy_score(labels_test, clf.predict(features_test))
print(f'{acc}')