# https://www.kdnuggets.com/2020/01/decision-tree-algorithm-explained.html

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
from sklearn.metrics import plot_roc_curve

#import Dataset
# df = pd.read_csv("../alzheimer/alzheimer.csv")
df = pd.read_csv("../alzheimer/Copy_alz.csv")


# get the matrix from Dataframe

df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})        # Male is 1, female is 0
df['Group'] = df['Group'].map({'Demented': 1, 'Converted':1, 'Nondemented': 0})     # Demented and converted is 1, Nondemented is 0

# replace
df.fillna(0, inplace=True)
print(df.isna().sum())

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

# # # separate the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 1)

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()  
x_train= st_x.fit_transform(X_train)    
x_test= st_x.transform(X_test)    


#Fitting Decision Tree classifier to the training set  
from sklearn.tree import DecisionTreeClassifier  
classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)  
classifier.fit(x_train, y_train)  

#Predicting the test set result  
y_pred= classifier.predict(x_test)  


# Accuracy Score 
# How often is the classifier correct?
acc =  classifier.score(x_test,y_test)

# Precicion score
# When it predicts yes, how often is it correct?
precision = precision_score(y_test,y_pred)

# Recall/sensitivity
# When it’s actually yes, how often does it predict yes?
recall = recall_score(y_test,y_pred)

# f1 score
# 
f1 = f1_score(y_test,y_pred)

# # Receiver Operating Characteristics (ROC) Curve
# # plotting TPR(sensitivity) vs FPR(1 — specificity), we get Receiver Operating Characteristic (ROC) curve

# #plotting roc curve
# plot_roc_curve(LogReg_clf,X_test,y_test)
# plt.plot([0,1],[0,1],'k--')
# plt.title('ROC Curve and AUC')
# # plt.legend()
# plt.show()

# from sklearn.metrics import log_loss

# accuracy = log_loss(y_test, y_pred)
# print("Logloss: %.2f" % (accuracy))

print("Logistic Regression\n")
print("Accuracy Score", acc)
print("Precision Score",precision)
print("Recall score",recall)
print("f1_score",f1)

# Confusion MAtrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cm)


# from sklearn.linear_model import LassoCV

# reg = LassoCV()
# reg.fit(X, y)
# print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
# print("Best score using built-in LassoCV: %f" %reg.score(X,y))
# coef = pd.Series(reg.coef_, index = X1.columns)

# print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

# imp_coef = coef.sort_values()
# import matplotlib
# matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
# imp_coef.plot(kind = "barh")
# plt.title("Feature importance using Lasso Model")
# plt.show()
