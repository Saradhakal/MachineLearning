
#importing all the required ML packages
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
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
df.rename(columns={'M/F': 'Gender'}, inplace = True)

# replace
df.fillna(0, inplace=True)

# Separate input and output variables
X = df[['Gender', 'EDUC', 'SES', 'MMSE','CDR', 'eTIV']]
y = df['Group'].values

#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()  
x_scaled= st_x.fit_transform(X)    

# # # separate the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.33, random_state = 1)
# from sklearn.preprocessing import MinMaxScaler

# trans = MinMaxScaler()
# # mm_S = trans.fit_transform(data)
# x_train= trans.fit_transform(X_train)    
# x_test= trans.transform(X_test)    

# # convert the array back to a dataframe

# /
# error = []

# # Calculating error for K values between 1 and 30
# for i in range(1, 30):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(x_train, y_train)
#     pred_i = knn.predict(x_test)
#     error.append(np.mean(pred_i != y_test))
# plt.figure(figsize=(12, 6))
# plt.plot(range(1, 30), error, color='red', linestyle='dashed', marker='o',
#          markerfacecolor='blue', markersize=10)
# plt.title('Error Rate K Value')
# plt.xlabel('K Value')
# plt.ylabel('Mean Error')
# print("Minimum error:-",min(error),"at K =",error.index(min(error))+1)


# model
knn_classifier=KNeighborsClassifier(n_neighbors=5) 

# model.fit(X_train,y_train)
knn_classifier.fit(X_train,y_train)

y_pred=knn_classifier.predict(X_test)
print('The accuracy of the KNN is',metrics.accuracy_score(y_test,y_pred))

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

print("\n K N N\n")
# print("Accuracy Score", acc*100)
print("Precision Score",precision*100)
print("Recall score",recall*100)
print("f1_score",f1*100)
print('FPR', (1-recall_specificity)*100)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cm)

# a_index=list(range(1,11))
# a=pd.Series()
# x=[0,1,2,3,4,5,6,7,8,9,10]
# for i in list(range(1,11)):
#     model=KNeighborsClassifier(n_neighbors=i) 
#     model.fit(x_train,y_train)
#     prediction=model.predict(x_test)
#     a=a.append(pd.Series(metrics.accuracy_score(prediction,y_test)))
# plt.plot(a_index, a)
# plt.xticks(x)
# fig=plt.gcf()
# fig.set_size_inches(12,6)
# plt.show()
# print('Accuracies for different values of n are:',a.values,'with the max value as ',a.values.max())

# # evaluate adaboost algorithm for classification
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

n_scores = cross_val_score(knn_classifier, x_scaled, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# from sklearn.ensemble import BaggingClassifier
# model=BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=5),random_state=0,n_estimators=700)
# # model.fit(X_train,y_train)

# model.fit(x_train,y_train)
# prediction=model.predict(x_test)
# print('The accuracy for bagged KNN is:',metrics.accuracy_score(prediction,y_test))

# result=cross_val_score(model,X,y,cv=10,scoring='accuracy')
# print('The cross validated score for bagged KNN is:',result.mean())