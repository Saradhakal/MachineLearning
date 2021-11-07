import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score, precision_score, recall_score, f1_score
from sklearn import metrics #accuracy measure
from sklearn.metrics import roc_auc_score, roc_curve

import sklearn_relief as sr

#import Dataset
# df = pd.read_csv("../alzheimer/alzheimer.csv")
df = pd.read_csv("../alzheimer/Copy_alz.csv")

# get the matrix from Dataframe

df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})        # Male is 1, female is 0
df['Group'] = df['Group'].map({'Demented': 1, 'Converted':1, 'Nondemented': 0})     # Demented and converted is 1, Nondemented is 0

# replace
# df.fillna(0, inplace=True)
# print(df.isna().sum())

# # Replace
# calculate mean
column_means = df.mean()
# print(column_means)

# Replace
df = df.fillna(column_means)
# print(df.isnull().sum())

data = df.copy() # for VISUALIZATION
X = data.drop("Group",axis=1)
y = data["Group"]


fs = sr.RReliefF(n_features=3)
X_train = fs.fit_transform(X, y)
print("(No. of tuples, No. of Columns before ReliefF) : "+str(X.shape)+
      "\n(No. of tuples, No. of Columns after ReliefF) : "+str(X_train.shape))