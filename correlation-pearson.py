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


# data = df.to_numpy()
# print(data)

# X1= df.drop("Group",1)
# Separate input and output variables
# X, y = data[:, 1:], data[:,0]

# # # separate the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 1)

# #Using Pearson Correlation
# plt.figure(figsize=(12,10))
# cor = df.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.show()

# #Correlation with output variable
# cor_target = abs(cor["Group"])

# #Selecting highly correlated features
# relevant_features = cor_target[cor_target>0.0]
# print(relevant_features)

# from sklearn.feature_selection import RFE
# from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso


# Create correlation matrix
corr_matrix = df.corr()
print(corr_matrix)

# Create correlation heatmap
plt.figure(figsize=(8,6))
plt.title('Correlation Heatmap of OASIS Dataset')
a = sns.heatmap(corr_matrix, square=True, annot=True, fmt='.2f', linecolor='black')
a.set_xticklabels(a.get_xticklabels(), rotation=30)
a.set_yticklabels(a.get_yticklabels(), rotation=30)           
plt.show()    

 # Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# print(upper)
# print(upper.info())


to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
print('drop',to_drop)

# Drop Marked Features
df1 = df.drop(df.columns[to_drop], axis=1)
print(df1)

