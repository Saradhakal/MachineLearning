import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
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

#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()  
x_train= st_x.fit_transform(X_train)    
x_test= st_x.transform(X_test)    

from sklearn.ensemble import AdaBoostClassifier
Model=AdaBoostClassifier()
Model.fit(x_train, y_train)
y_pred=Model.predict(x_test)

# Summary of the predictions made by the classifier

# Accuracy score
acc = accuracy_score(y_pred,y_test)

# Precicion score
# When it predicts yes, how often is it correct?
precision = precision_score(y_test,y_pred)

# Recall/sensitivity
# When it’s actually yes, how often does it predict yes?
recall = recall_score(y_test,y_pred)

# f1 score
f1 = f1_score(y_test,y_pred)

print("\nAda Boost\n")
print("Accuracy Score", acc)
print("Precision Score",precision)
print("Recall score",recall)
print("f1_score",f1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cm)


rf_adaboost_model = AdaBoostClassifier(base_estimator=AdaBoostClassifier(), n_estimators=100, learning_rate=0.1, algorithm='SAMME.R', random_state=0)

rf_adaboost_model.fit(x_train , y_train)
rf_adaboost_train_score = rf_adaboost_model.score(x_train , y_train)

print("Adaboost AdaBoost Classifier Training Score: {:.3F}".format(rf_adaboost_train_score))


# # evaluate adaboost algorithm for classification
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

# AdaBoostClassifier evaluated using shuffle-split cross-validation 
rf_adaboost_shuffle_split = StratifiedShuffleSplit(train_size=0.8, test_size=0.2, n_splits=5, random_state=0)
rf_adaboost_val_scores = cross_val_score(rf_adaboost_model, x_train , y_train, cv=rf_adaboost_shuffle_split)
print("Adaboost AdaBoost Classifier Cross validation Score: {:.3F}".format(np.mean(rf_adaboost_val_scores)))

# # evaluate adaboost algorithm for classification
# from numpy import mean
# from numpy import std
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold

# # evaluate the model
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# n_scores = cross_val_score(Model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# # report performance
# print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# from sklearn.ensemble import BaggingClassifier
# model=BaggingClassifier(base_estimator=RepeatedStratifiedKFold(n_neighbors=5),random_state=0,n_estimators=700)
# # model.fit(X_train,y_train)

# model.fit(x_train,y_train)
# prediction=model.predict(x_test)
# print('The accuracy for bagged KNN is:',metrics.accuracy_score(prediction,y_test))

# # # result=cross_val_score(model,X,Y,cv=10,scoring='accuracy')
# # # print('The cross validated score for bagged KNN is:',result.mean())