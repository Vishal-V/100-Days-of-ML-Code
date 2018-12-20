import numpy as np

def sigmoid(x):
	return 1.0 / (1 + np.exp(-x))

def y_hat(weights, bias, x):
	return np.dot(weights, x) + bias

def cost(y, output):
	return -(y*np.log(output) - (1-y)*np.log(1-output))

def gradient_descent(x, y, weights, bias, learnrate):
	y_h = y_hat(weights, bias, x)
	weights += learnrate * (y-y_h) * x
	bias += learnrate * (y-y_h)
	return weights, bias