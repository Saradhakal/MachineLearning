
# https://medium.com/@sabarirajan.kumarappan/feature-selection-by-lasso-and-ridge-regression-python-code-examples-1e8ab451b94b

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
import seaborn as sns

from sklearn.metrics import precision_recall_fscore_support as score, precision_score, recall_score, f1_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel


# import dataframe
# df = pd.read_csv("../alzheimer/alzheimer.csv")
df = pd.read_csv("../alzheimer/Copy_alz.csv")

df.head()
# print(df.shape)

# checking missing values in each column
# print(df.isnull().sum())

# SES and MMSE have null values , Replace with Mean

# calculate mean
column_means = df.mean()
# print(column_means)

# Replace
df = df.fillna(column_means)
# print(df.isnull().sum())

df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})        # Male is 1, female is 0
df['Group'] = df['Group'].map({'Demented': 1, 'Converted':1, 'Nondemented': 0})     # Demented and converted is 1, Nondemented is 0
df.rename(columns={'M/F': 'Gender'}, inplace = True)


# y is label and X are features
x = df[['Gender', 'Age-classification', 'EDUC', 'SES', 'MMSE','CDR', 'eTIV', 'nWBV', 'ASF']]
y = df['Group'].values


from sklearn.preprocessing import StandardScaler

# Scaling
scaler = StandardScaler()
X = scaler.fit(x)
Y = scaler.fit(y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state = 1)


# Selecting features using Lasso regularisation

sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))
sel_.fit(X, np.ravel(y,order='C'))
# print(sel_.get_support())
X_train = pd.DataFrame(X_train)

selected_feat = X_train.columns[(sel_.get_support())]
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
np.sum(sel_.estimator_.coef_ == 0)))

removed_feats = X_train.columns[(sel_.estimator_.coef_ == 0).ravel().tolist()]
print('Removed Features',removed_feats)
# X_selected = sel_.transform(X)
X_train_selected = sel_.transform(X_train)
X_test_selected = sel_.transform(X_test)
# X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.33, random_state = 1)

from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
Model=AdaBoostClassifier()
Model.fit(X_train,np.ravel(y_train,order='C'))

y_pred=Model.predict(X_train_selected)
# print(y_pred.shape)
# Summary of the predictions made by the classifier

# Accuracy score
acc = accuracy_score(y_pred,y_test)

# # Precicion score
# # When it predicts yes, how often is it correct?
precision = precision_score(y_test,y_pred)

# # Recall/sensitivity
# # When it’s actually yes, how often does it predict yes?
recall = recall_score(y_test,y_pred)

# f1 score
f1 = f1_score(y_test,y_pred)

print("\nAda Boost\n")
print("Accuracy Score", acc)
print("Precision Score",precision)
print("Recall score",recall)
print("f1_score",f1)


# from sklearn.ensemble import RandomForestClassifier
# # Create a random forest classifier
# clf = RandomForestClassifier(random_state=1, n_jobs=-1, max_depth=5, n_estimators=100, oob_score=True)

# clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
# # Train the classifier
# clf.fit(X_train,np.ravel(y_train,order='C'))
# # Apply The Full Featured Classifier To The Test Data
# y_pred = clf.predict(X_test)
# # View The Accuracy Of Our Selected Feature Model
# print(accuracy_score(y_test, y_pred))
# # # Precicion score
# # # When it predicts yes, how often is it correct?
# precision = precision_score(y_test,y_pred)

# # # Recall/sensitivity
# # # When it’s actually yes, how often does it predict yes?
# recall = recall_score(y_test,y_pred)

# # f1 score
# f1 = f1_score(y_test,y_pred)

# print("\Random Forest\n")
# print("Accuracy Score", acc)
# print("Precision Score",precision)
# print("Recall score",recall)
# print("f1_score",f1)

