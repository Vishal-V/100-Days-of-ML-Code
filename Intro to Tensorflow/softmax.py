# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 09:34:10 2018

@author: Vishal

Softmax function
"""

import numpy as np
scores = np.array([3.0, 1.0, 0.2])
scores = scores / 10.0


def softmax(x):
    #Compute softmax values for each sets of scores in x.
    return np.exp(x) / np.sum(np.exp(x), axis=0)

print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
