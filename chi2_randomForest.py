import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score, precision_score, recall_score, f1_score

#import Dataset
# df = pd.read_csv("../alzheimer/alzheimer.csv")
df = pd.read_csv("../alzheimer/Copy_alz.csv")


# get the matrix from Dataframe

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

# data = df.copy() # for VISUALIZATION
# X = data.drop("Group",axis=1)
# y = data["Group"]

# data = df.to_numpy()
# print(data)

# label encoding
df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})        # Male is 1, female is 0
df['Group'] = df['Group'].map({'Demented': 1, 'Converted':1, 'Nondemented': 0})     # Demented and converted is 1, Nondemented is 0
df.rename(columns={'M/F': 'Gender'}, inplace = True)


# Separate input and output variables
X = df[['Gender', 'EDUC', 'SES', 'MMSE','CDR', 'eTIV']]
y = df['Group'].values

#feature Scaling  
from sklearn.preprocessing import StandardScaler    

st_x= StandardScaler().fit(X)
x_scaled = st_x.transform(X)


# # # separate the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.33, random_state = 1)


# from sklearn.ensemble import RandomForestClassifier
# training the model on training set
rf_classifier = RandomForestClassifier(random_state=1, n_jobs=-1, max_depth=5, n_estimators=100, oob_score=True)
rf_classifier.fit(X_train, y_train)


# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(classifier_rf, X_train, y_train, cv=10, scoring = "accuracy")
# print("Scores:", scores)
# print("Mean:", scores.mean())
# print("Standard Deviation:", scores.std())
# # evaluate the model
y_pred = rf_classifier.predict(X_test)
# print(y_pred)

# Accuracy Score 
# How often is the classifier correct?
# acc =  rf_classifier.score(X_test,y_test)
acc = accuracy_score(y_pred,y_test)

# Precicion score
# When it predicts yes, how often is it correct?
precision = precision_score(y_test,y_pred)

# Recall/sensitivity
# When it???s actually yes, how often does it predict yes?
recall = recall_score(y_test,y_pred)

# f1 score
f1 = f1_score(y_test,y_pred)

# recall_sensitivity = metrics.recall_score(y_test, y_pred, pos_label=1)
recall_specificity = (metrics.recall_score(y_test, y_pred, pos_label=0))

print("\n Random Forest \n")
print("Accuracy Score", acc*100)
print("Precision Score",precision*100)
print("Recall score",recall*100)
print("f1_score",f1*100)
print('FPR', (1-recall_specificity)*100)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cm)

# # evaluate adaboost algorithm for classification
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import KFold

# evaluate the model
# cv = RepeatedStratifiedKFold(n_splits=10,random_state=1)
cv = KFold(n_splits=10,random_state=1,shuffle=True)
n_scores = cross_val_score(rf_classifier, x_scaled, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# report performance
print('Accuracy: %.4f (%.4f)' % (mean(n_scores), std(n_scores)))

# Bagging
# 
# from sklearn.ensemble import BaggingClassifier
# model = BaggingClassifier(base_estimator=RandomForestClassifier(random_state=1),random_state=0,n_estimators=100)

# model.fit(x_train,y_train)
# prediction = model.predict(x_test)
# rf_adaboost_train_score = model.score(x_train , y_train)
# print('RandomForest Bagging Classifier Training Score:',metrics.accuracy_score(prediction,y_test))