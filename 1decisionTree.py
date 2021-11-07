# https://www.kdnuggets.com/2020/01/decision-tree-algorithm-explained.html

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score, precision_score, recall_score, f1_score

from sklearn import metrics #accuracy measure


#import Dataset
# df = pd.read_csv("../alzheimer/alzheimer.csv")
df = pd.read_csv("../alzheimer/Copy_alz.csv")


# get the matrix from Dataframe

df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})        # Male is 1, female is 0
df['Group'] = df['Group'].map({'Demented': 1, 'Converted':1, 'Nondemented': 0})     # Demented and converted is 1, Nondemented is 0

# replace
df.fillna(0, inplace=True)
# print(df.isna().sum())

# column_means = df.mean()
# df = df.fillna(column_means)

# data = df.copy() # for VISUALIZATION
# X = data.drop("Group",axis=1)
# y = data["Group"]

data = df.to_numpy()
# print(data)

# X1= df.drop("Group",1)
# Separate input and output variables
X, y = data[:, 1:], data[:,0]

#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()  
x_scaled= st_x.fit_transform(X)    

# # # separate the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.33, random_state = 1)


#Fitting Decision Tree classifier to the training set  
from sklearn import tree 
dt_classifier= DecisionTreeClassifier(criterion='entropy', random_state=1)  

dt_classifier.fit(X_train, y_train)  

#Predicting the test set result  
y_pred= dt_classifier.predict(X_test)  

# Accuracy Score 
# How often is the classifier correct?
# acc =  classifier.score(x_test,y_test)
acc = accuracy_score(y_pred,y_test)

# Precicion score
# When it predicts yes, how often is it correct?
precision = precision_score(y_test,y_pred)

# Recall/sensitivity
# When it’s actually yes, how often does it predict yes?
recall = recall_score(y_test,y_pred)

# f1 score
f1 = f1_score(y_test,y_pred)

# recall_sensitivity = metrics.recall_score(y_test, y_pred, pos_label=1)
recall_specificity = (metrics.recall_score(y_test, y_pred, pos_label=0))

print("\n Decision Tree\n")
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
from sklearn.model_selection import RepeatedStratifiedKFold


from sklearn.model_selection import KFold

# evaluate the model
# cv = RepeatedStratifiedKFold(n_splits=10,random_state=1)
cv = KFold(n_splits=10,random_state=1,shuffle=True)

# evaluate the model
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

n_scores = cross_val_score(dt_classifier, x_scaled, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# from sklearn.ensemble import BaggingClassifier
# model=BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', random_state=1),random_state=1,n_estimators=100)
# model.fit(X_train,y_train)
# prediction=model.predict(X_test)
# print('The accuracy for bagged Decision Tree is:',accuracy_score(prediction,y_test))
