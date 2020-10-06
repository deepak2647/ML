# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 17:03:12 2020

@author: Deepak
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the datasets
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#Training the linear regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Training the Polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visualizing the results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('truth or bluff (Linear regression)')
plt.xlabel('position of work')
plt.ylabel('salary range')
plt.show()

#Visualizing polynomial regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('truth or bluff (Polynomial regression)')
plt.xlabel('position of work')
plt.ylabel('salary range')
plt.show()

#visualizing polynomial regression results with smoother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('truth or bluff (Polynomial regression)')
plt.xlabel('position of work')
plt.ylabel('salary range')
plt.show()
 
#predicting the results
lin_reg.predict([[6.5]])

#Predicting the polynomial regression results
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))