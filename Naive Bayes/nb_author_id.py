#!/usr/bin/python

""" 
    Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB

"""
The pickle file has to be using Unix new lines otherwise at least Python 3.4's C 
pickle parser fails with exception: pickle.UnpicklingError: the STRING opcode 
argument must be quoted. I think that some git versions may be changing the 
Unix new lines ('\n') to DOS lines ('\r\n').
"""

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################

nb_clf = GaussianNB()
# Training the classifier with the training data
t = time()
nb_clf.fit(features_train, features_test)
print(f'Time to train : {round(time() - t, 4)}')

pred = nb_clf.predict(features_train)

X = features_test
Y = labels_test
accuracy = nb_clf.score(X,Y)
print(f'The accuracy is: {accuracy}')
#########################################################


