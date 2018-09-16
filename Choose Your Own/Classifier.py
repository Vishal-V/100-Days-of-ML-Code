from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from time import time

iris = datasets.load_iris()
X = iris.data
Y = iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

clf = KNeighborsClassifier(n_neighbors=10)
t = time()
clf.fit(X_train, Y_train)
print "Time to train: ", round(time() - t,3)
t = time()
pred = clf.predict(X_test)
print "Time to test: ", round(time() - t,3)
acc = accuracy_score(pred, Y_test)
print "Accuracy: " +  str(acc)

"""
    Time to train:  0.0
    Time to test:  0.002
    Accuracy: 1.0
    
"""