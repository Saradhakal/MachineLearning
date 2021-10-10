import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

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
# print(y)

# # Descriptive analysis of numerical values
# # print(df.info())
# # print(df.describe(include='all'))  

data = df.to_numpy()
# print(data)

# Separate input and output variables
X, y = data[:, 1:], data[:,0]

# # # separate the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 1)

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# # evaluate the model
y_hat = model.predict(X_test)
# print(y_hat)

# # evaluate predictions
mae = mean_absolute_error(y_test, y_hat)
print('MAE: %.3f' % mae)

# plotting residual errors in training data
plt.scatter(model.predict(X_train),model.predict(X_train) - y_train, color = "green", s= 10, label = 'Train data')

# plotting residual errors in test data
plt.scatter(model.predict(X_test),model.predict(X_test) - y_test, color = "blue", s= 10, label = 'Test data')

# plotting line for zero residual error
plt.hlines(y=0, xmin = 0, xmax= 50, linewidth = 2)

# plotting legends
plt.legend(loc= 'upper right')

# plot title
plt.title("Residual errors")

# method call for showing the plot
plt.show()

# variance score: 1 means perfect prediction
print('Variance score: {}'.format(model.score(X_test, y_test)))
