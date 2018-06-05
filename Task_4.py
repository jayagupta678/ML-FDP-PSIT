# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 13:10:15 2017

@author: JG
"""
'''#####################################
         Gaussian Naive Bayes Classifier
   #####################################'''

import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
# here, we remove the first example of each flower
# found at indices: 0, 50, 100
test_idx = [0, 50, 100]

# create 2 new sets of variables, for training and testing
# training data
# remove the entires from the data and target variables
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]
print(test_target)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb = gnb.fit(train_data, train_target)
predictions = gnb.predict(test_data)
print(predictions)

