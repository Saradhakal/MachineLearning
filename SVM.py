import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix



#import Dataset
df = pd.read_csv("../alzheimer/alzheimer.csv")

df.head()

# get the matrix from Dataframe
# print(df['M/F'])
# print(df['Group'])
df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})
df['Group'] = df['Group'].map({'Demented': 1, 'Nondemented': 0})
# print(df['M/F'])

# replace
df.fillna(0, inplace=True)
print(df.isna().sum())

column_means = df.mean()
df = df.fillna(column_means)


data = df.to_numpy()
# print(data)

# Separate input and output variables
X, y = data[:, 1:], data[:,0]
# print(X)

# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

# svm
svclassifier = SVC(kernel = 'linear')
svclassifier.fit(X_train, y_train)

# predict
y_pred = svclassifier.predict(X_test)

# Evaluate
print ("SVM \n")
print("confusion matrix \n")
print(confusion_matrix(y_test, y_pred))

print("class report \n")
print(classification_report(y_test, y_pred))