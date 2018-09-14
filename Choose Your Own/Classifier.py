from sklearn import discriminant_analysis
from sklearn import tree
from sklearn import neighbors

x = [[9,9,9], [6,8,9], [9,8,7], [6,9,7], [9,9,9], [6,8,9], [9,8,7], [6,9,7]]
y = ['Elon Musk', 'Steve Jobs', 'Peter Thiel', 'Mark Cuban', 'Elon Musk', 'Steve Jobs', 'Peter Thiel', 'Mark Cuban']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x,y)

clf1 = discriminant_analysis.QuadraticDiscriminantAnalysis()
clf1 = clf1.fit(x,y)

clf2 = neighbors.KNeighborsClassifier(n_neighbors = 3)
clf2 = clf2.fit(x,y)

prediction = clf.predict([[5,2,0]])
prediction1 = clf1.predict([[5,2,0]])
prediction2 = clf2.predict([[5,2,0]])

print(prediction)
print(prediction1)
print(prediction2)
