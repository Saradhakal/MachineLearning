import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import svm
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score, precision_score, recall_score, f1_score

#import Dataset
# df = pd.read_csv("../alzheimer/alzheimer.csv")
df = pd.read_csv("../alzheimer/Copy_alz.csv")

# get the matrix from Dataframe

df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})        # Male is 1, female is 0
df['Group'] = df['Group'].map({'Demented': 1, 'Converted':1, 'Nondemented': 0})     # Demented and converted is 1, Nondemented is 0

# Find na
# df.fillna(0, inplace=True)
# print(df.isna().sum())

# calculate mean
column_means = df.mean()
# print(column_means)

# Replace
df = df.fillna(column_means)
# print(df.isnull().sum())

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
# print(X)

# #feature Scaling  
# from sklearn.preprocessing import StandardScaler    

# st_x= StandardScaler().fit(X,y)
# X_scaled = st_x.transform(X)
# # x_test = st_x.transform(X_test)


# # # separate the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 1)


# scaler = StandardScaler().fit(xTrain, yTrain)
# xRTrain = scaler.transform(xTrain)
# xRTest = scaler.transform(xTest)

# svm
svclassifier = svm.SVC(kernel = 'linear')
svclassifier.fit(X_train, y_train)

# predict
y_pred = svclassifier.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
acc=metrics.accuracy_score(y_test, y_pred) 

        # print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")

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

print("\n S V M\n")
print("Accuracy Score", acc*100)
print("Precision Score",precision*100)
print("Recall score",recall*100)
print("f1_score",f1*100)
print('FPR', (1-recall_specificity)*100)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cm)
# print("class report \n")
# print(classification_report(y_test, y_pred))


#  evaluate algorithm for classification
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import KFold

# evaluate the model
# cv = RepeatedStratifiedKFold(n_splits=10,random_state=1)
cv = KFold(n_splits=10,random_state=1,shuffle=True)

n_scores = cross_val_score(svclassifier, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# report performance
print('Accuracy: %.4f (%.4f)' % (mean(n_scores), std(n_scores)))



# from sklearn.model_selection import GridSearchCV

# # defining parameter range 
# param_grid = {'C': [0.1, 1, 10, 100],'gamma': [1, 0.1, 0.01, 0.001, 0.0001],'gamma':['scale', 'auto'], 'kernel': ['linear']}  
   
# grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3,n_jobs=-1) 
   
# # fitting the model for grid search 
# grid.fit(X_train, y_train) 
 
# # print best parameter after tuning 
# print(grid.best_params_) 
# grid_predictions = grid.predict(X_test)    
# # print classification report 
# acc=metrics.accuracy_score(y_test, grid_predictions)
# print('acc', acc)
# # print(classification_report(y_test, grid_predictions)) 