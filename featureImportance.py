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

# #feature Scaling  
from sklearn.preprocessing import StandardScaler   
from sklearn.preprocessing import MinMaxScaler

# st_x= StandardScaler()  
st_x= MinMaxScaler()  

x_train= st_x.fit_transform(X_train)    
x_test= st_x.transform(X_test)  

x_train = pd.DataFrame(x_train)

model = RandomForestClassifier(n_estimators=100, random_state=10, n_jobs = -1)

model.fit(x_train, y_train)

# features = list(x_train.columns)
# # Feature importances into a dataframe
# feature_importances = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
# print(feature_importances.head() )                              



importances= model.feature_importances_
feature_importances= pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
sns.barplot(x=feature_importances[0:10], y=feature_importances.index[0:10], palette="rocket")
sns.despine()
plt.xlabel("Feature Importances")
plt.ylabel("Features")
plt.show()