# https://www.kdnuggets.com/2020/05/model-evaluation-metrics-machine-learning.html

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import metrics #accuracy measure


#import Dataset
# df = pd.read_csv("../alzheimer/alzheimer.csv")
df = pd.read_csv("../alzheimer/Copy_alz.csv")


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


df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})        # Male is 1, female is 0
df['Group'] = df['Group'].map({'Demented': 1, 'Converted':1, 'Nondemented': 0})     # Demented and converted is 1, Nondemented is 0

data = df.to_numpy()
# print(data)

data = df.copy() # for VISUALIZATION
# X = data.drop("Group",axis=1)
# y = data["Group"]

# data = df.to_numpy()
# print(data)

# # X1= df.drop("Group",1)
# # Separate input and output variables
# X, y = data[:, 1:], data[:,0]

from sklearn.preprocessing import StandardScaler    
scaler = StandardScaler()
# transform data
scaled = scaler.fit_transform(data)
X = data.drop("Group",axis=1)
y = data["Group"]
# Separate input and output variables
# X, y = data[:, 1:], data[:,0]

# #feature Scaling  
# from sklearn.preprocessing import StandardScaler    

# st_x= StandardScaler().fit(X,y)
# X_scaled = st_x.transform(X)

# # # separate the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 1)

# fit the model
LogReg_clf = LogisticRegression(random_state = None, max_iter=100)
LogReg_clf.fit(X_train, y_train)

# # evaluate the model
y_pred = LogReg_clf.predict(X_test)
# print(y_hat)

# Accuracy Score 
# How often is the classifier correct?
# acc =  LogReg_clf.score(X_test,y_test)
acc = accuracy_score(y_pred,y_test)

# print('Root mean squared error',np.sqrt(mean_squared_error(y_test,y_pred))) 

# Precicion score
# When it predicts yes, how often is it correct?
precision = precision_score(y_test,y_pred)

# Recall/sensitivity
# When it’s actually yes, how often does it predict yes?
recall = recall_score(y_test,y_pred)

# f1 score
f1 = f1_score(y_test,y_pred)

# # Receiver Operating Characteristics (ROC) Curve
# # plotting TPR(sensitivity) vs FPR(1 — specificity), we get Receiver Operating Characteristic (ROC) curve

# # predict probabilities
# probs = LogReg_clf.predict_proba(X_test)
# # keep probabilities for the positive outcome only
# probs = probs[:, 1]

# auc = roc_auc_score(y_test, probs)
# print('AUC - Test Set: %.2f%%' % (auc*100))

# # calculate roc curve
# fpr, tpr, thresholds = roc_curve(y_test, probs)
# # plot no skill
# plt.plot([0, 1], [0, 1], linestyle='--')
# # plot the roc curve for the model
# plt.plot(fpr, tpr, marker='.')
# plt.xlabel('False positive rate')
# plt.ylabel('Sensitivity/ Recall')
# # show the plot
# plt.show()

# from sklearn.metrics import log_loss

# accuracy = log_loss(y_test, y_pred)
# print("Logloss: %.2f" % (accuracy))


# recall_sensitivity = metrics.recall_score(y_test, y_pred, pos_label=1)
recall_specificity = (metrics.recall_score(y_test, y_pred, pos_label=0))

print("\n Logistic Regression\n")
print("Accuracy Score", acc*100)
print("Precision Score",precision*100)
print("Recall score",recall*100)
print("f1_score",f1*100)
print('FPR', (1-recall_specificity)*100)
# Confusion MAtrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cm)


# evaluate adaboost algorithm for classification
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# evaluate the model
# cv = RepeatedStratifiedKFold(n_splits=10,random_state=1)
cv = KFold(n_splits=10, random_state=1, shuffle=True)
# evaluate the model
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

n_scores = cross_val_score(LogReg_clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
