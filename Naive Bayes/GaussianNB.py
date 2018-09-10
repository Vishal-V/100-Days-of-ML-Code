"""
Created on Sun Sep  9 20:39:19 2018
@author: Vishal V

"""
#Using the iris dataset from sklearn
from sklearn import datasets
#Using a Gaussian Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
iris = datasets.load_iris()
clf = GaussianNB()
X = iris.data
Y = iris.target
pred = clf.fit(X, Y).predict(X)
size = X.shape[0]
value = (Y != pred).sum()
print(f'The number of mislabeled points from {size} are: {value}')
accuracy = 1 - ((Y != pred).sum() / len(pred))
acc = clf.score(X,Y)
print(f'The accuracies are {accuracy} and {acc}')