import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pandas import read_csv
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import mean_absolute_error
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score, precision_score, recall_score, f1_score
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

# data = df.copy() # for VISUALIZATION
# X = data.drop("Group",axis=1)
# y = data["Group"]

data = df.to_numpy()
# print(data)

# X1= df.drop("Group",1)
# Separate input and output variables
X, y = data[:, 1:], data[:,0]

# # # separate the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 1)

# #feature Scaling  
# from sklearn.preprocessing import StandardScaler    
# st_x= StandardScaler()  
# x_train= st_x.fit_transform(X_train)    
# x_test= st_x.transform(X_test)    


from sklearn.preprocessing import MinMaxScaler
trans = MinMaxScaler()
x_train = trans.fit_transform(X_train)
x_test= trans.transform(X_test)    

# from sklearn.ensemble import RandomForestClassifier
# training the model on training set
rf_classifier = RandomForestClassifier(random_state=1, n_jobs=-1, max_depth=5, n_estimators=100, oob_score=True)
rf_classifier.fit(x_train, y_train)


# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(classifier_rf, X_train, y_train, cv=10, scoring = "accuracy")
# print("Scores:", scores)
# print("Mean:", scores.mean())
# print("Standard Deviation:", scores.std())
# # evaluate the model
y_pred = rf_classifier.predict(x_test)
# print(y_pred)

# Accuracy Score 
# How often is the classifier correct?
# acc =  rf_classifier.score(X_test,y_test)
acc = accuracy_score(y_pred,y_test)

# Precicion score
# When it predicts yes, how often is it correct?
precision = precision_score(y_test,y_pred)

# Recall/sensitivity
# When it’s actually yes, how often does it predict yes?
recall = recall_score(y_test,y_pred)

# f1 score
f1 = f1_score(y_test,y_pred)

# Confusion MAtrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cm)


rf_bagging_model = BaggingClassifier(base_estimator=RandomForestClassifier(),n_estimators=100, bootstrap=True, max_samples=100, n_jobs=-1, random_state=0)

rf_bagging_model.fit(x_train , y_train)
rf_bagging_train_score = rf_bagging_model.score(x_train , y_train)

print("RandomForest Bagging Classifier Training Score: {:.3F}".format(rf_bagging_train_score))

# # evaluate adaboost algorithm for classification
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

#  BaggingClassifier evaluated using shuffle-split cross-validation 
rf_bagging_shuffle_split = StratifiedShuffleSplit(train_size=0.8, test_size=0.2, n_splits=5, random_state=0)
rf_baggign_val_scores = cross_val_score(rf_bagging_model, x_train , y_train, cv=rf_bagging_shuffle_split)
print("RandomForest Bagging Classifier Cross validation Score: {:.3F}".format(np.mean(rf_baggign_val_scores)))