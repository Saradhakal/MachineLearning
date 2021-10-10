import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pandas import read_csv
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#import Dataset
df = pd.read_csv("../alzheimer/alzheimer.csv")

# get the matrix from Dataframe
print(df['M/F'])
print(df['Group'])
df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})
df['Group'] = df['Group'].map({'Demented': 1, 'Nondemented': 0})
print(df['M/F'])

# replace
df.fillna(0, inplace=True)
print(df.isna().sum())

column_means = df.mean()
df = df.fillna(column_means)

# data = df.copy() # for VISUALIZATION
# X = data.drop("Group",axis=1)
# y = data["Group"]

# print(X)
# print(y)

# # Descriptive analysis of numerical values
# # print(df.info())
# # print(df.describe(include='all'))  

data = df.to_numpy()
# print(data)

# # Separate input and output variables
X, y = data[:, 1:], data[:,0]

# # # separate the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 1)

# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)

# identify outliers in the training dataset
# fit_predict returns labels for each record in training set
# (-1: outliers, 1: inliers)
iso = IsolationForest(contamination=0.1)
y_hat = iso.fit_predict(X_train)

# select all rows that are not outliers
mask = y_hat != -1
X_train, y_train = X_train[mask, :], y_train[mask]


# fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# # evaluate the model
y_hat = model.predict(X_test)
# print(y_hat)

# # evaluate predictions
mae = mean_absolute_error(y_test, y_hat)
print('MAE: %.3f' % mae)

# # accuracy
# print("Accuracy: " % (metrics.accuracy_score(y_test, y_hat)*100))

# # confusion matrix
# cm = confusion_matrix(y_test, y_hat, labels= model.classes)

# #cm as heatmap
# disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=datasets.target_names)
# disp.plot()
# plt.show()
