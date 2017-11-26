# -*- coding: utf-8 -*-
# Data Preprocessing

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the data set
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# cleaning up missing data by filling missing entries with the mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding Caregorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
ohe = OneHotEncoder(categorical_features=[0])
X = ohe.fit_transform(X).toarray()
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)