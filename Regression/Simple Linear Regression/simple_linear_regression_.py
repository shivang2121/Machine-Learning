# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 09:48:42 2021

@author: PS
"""
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset
data=pd.read_csv("Salary_Data.csv")
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


#Split into test and train data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Train the model and predict on test set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)

#visualize on training set
plt.scatter(X_train,y_train, color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title("Salary vs experience (training set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

#visualize on test set
plt.scatter(X_test,y_test, color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title("Salary vs experience (test set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()