# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 23:39:02 2021

@author: PS
"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset
data=pd.read_csv("Position_Salaries.csv")
X=data.iloc[:,1:-1].values
y=data.iloc[:,-1].values

#build linear regressor
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

#linear polynomial model
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# Visualising the polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))
#print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))