# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Import Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
np.set_printoptions(threshold=np.nan)


#Importing DataSet 
dataset = pd.read_csv(r"E:\jaya\ML and NN fdp data\my handson\Tasks\kc_house_data.csv")
space=dataset['sqft_living']
price=dataset['price']

x = np.array(space).reshape(-1, 1)
y = np.array(price)

#Splitting the data into Train and Test
from sklearn.cross_validation import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=1/3, random_state=0)


#Fitting simple linear regression to the Training Set
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)


#Visualizing the training Test Results 
plt.scatter(xtrain, ytrain, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

#Predicting the prices
pred = regressor.predict(xtest)

print(regressor.predict(318880))
#Visualizing the Test Results 
plt.scatter(xtest, ytest, color= 'red')
plt.plot(xtest, regressor.predict(xtest), color = 'blue')
plt.title("Visuals for Test DataSet")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()