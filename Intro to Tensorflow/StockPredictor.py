import numpy
import csv
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []

def get_data(filename):
	with open(filename, 'r') as file:
		csvReader = csv.reader(file)
		next(csvReader)
		for row in csvReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(row[1])
	return

def predict_prices(dates, prices):
	dates = numpy.reshape(dates, len(dates), 1)

	svr_lin = SVR(kernel='linear', c=1e3)
	svr_poly = SVR(kernel='poly', c=1e3, degree=2)
	svr_rbf = SVR(kernel='rbf', c=1e3, gamma=0.01)
	svr_lin.fit(dates,prices)
	svr_poly.fit(dates,prices)
	svr_rbf.fit(dates,prices)

	plt.scatter(dates, prices, color= 'black', label= 'Data') 
	plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') 
	plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model')
	plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') 
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()

	return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_pol.predict(x)[0]

get_data('aapl.csv')

predict_prices(dates, prices)


