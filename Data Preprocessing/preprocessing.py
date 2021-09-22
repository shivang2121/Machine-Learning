
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset
data=pd.read_csv("Data.csv")
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

# Taking care of missing data 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Onehotencoding to avoid order while encoding as number (i,e.. 001 instead of 0 and 010 instead)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X=np.array(ct.fit_transform(X))

#label encoder for 0 as no and 1 as yes
from sklearn.preprocessing import LabelEncoder
ct=LabelEncoder()
y=ct.fit_transform(y)

#test train split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#standard scaler
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train[:,3:]=sc.fit_transform(X_train[:,3:])
X_test[:,3:]=sc.transform(X_test[:,3:])