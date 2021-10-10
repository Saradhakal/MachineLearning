
# https://medium.com/@sabarirajan.kumarappan/feature-selection-by-lasso-and-ridge-regression-python-code-examples-1e8ab451b94b

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
import seaborn as sns

# import dataframe
# df = pd.read_csv("../alzheimer/alzheimer.csv")
df = pd.read_csv("../alzheimer/Copy_alz.csv")

df.head()
print(df.shape)

# Convert into labels, 

df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})        # Male is 1, female is 0
df['Group'] = df['Group'].map({'Demented': 1, 'Converted':1, 'Nondemented': 0})     # Demented and converted is 1, Nondemented is 0

# y is label and X are features
y = df['Group'].values
X = df[['M/F', 'Age-classification', 'EDUC', 'SES', 'MMSE','CDR', 'eTIV', 'nWBV', 'ASF']]
# y = df['Group']
df.rename(columns={'M/F': 'Gender'}, inplace = True)

# print(y)
# print(X)

# checking missing values in each column
print(df.isnull().sum())

# SES and MMSE have null values , Replace with Mean

# calculate mean
column_means = df.mean()
print(column_means)

# Replace
df = df.fillna(column_means)
print(df.isnull().sum())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel

print(df.shape)
numerics = ['int16','int32','int64','float16','float32','float64']
numerical_vars = list(df.select_dtypes(include=numerics).columns)
data = df[numerical_vars]
print(data.shape)

data = df.to_numpy()
# print(data)

# X1= df.drop("Group",1)
# Separate input and output variables
X, y = data[:, 1:], data[:,0]


# # # separate the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 1)
# print((X_train.shape),(X_test.shape))

from sklearn.preprocessing import MinMaxScaler
Min_Max = MinMaxScaler()
Min_Max.fit_transform(X_train)
# Y= Min_Max.fit_transform(y)



from sklearn.preprocessing import StandardScaler

# # Scaling
# scaler = StandardScaler()
# scaler.fit(X_train)
# # print(X_train.shape)

# from sklearn.neighbors import KNeighborsClassifier

# # # train test split
# # from sklearn.model_selection import train_test_split
# # # x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
# # knn = KNeighborsClassifier(n_neighbors = 2)
# # # x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']
# # knn.fit(X_train,y_train)
# # prediction = knn.predict(X_test)
# # #print('Prediction: {}'.format(prediction))
# # print('With KNN (K=3) accuracy is: ',knn.score(X_test,y_test)) 
# # Model complexity

# neig = np.arange(1, 25)
# train_accuracy = []
# test_accuracy = []
# # Loop over different values of k
# for i, k in enumerate(neig):
#     # k from 1 to 25(exclude)
#     knn = KNeighborsClassifier(n_neighbors=k)
#     # Fit with knn
#     knn.fit(X_train,y_train)
#     #train accuracy
#     train_accuracy.append(knn.score(X_train, y_train))
#     # test accuracy
#     test_accuracy.append(knn.score(X_test, y_test))

# # Plot
# plt.figure(figsize=[13,8])
# plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
# plt.plot(neig, train_accuracy, label = 'Training Accuracy')
# plt.legend()
# plt.title('-value VS Accuracy')
# plt.xlabel('Number of Neighbors')
# plt.ylabel('Accuracy')
# plt.xticks(neig)
# plt.savefig('graph.png')
# plt.show()
# print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))


# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(random_state = 4)
# rfc=rf.fit(X_train,y_train)
# y_pred = rf.predict(X_test)
# cm = confusion_matrix(y_test,y_pred)
# print('Confusion matrix: \n',cm)
# print('Classification report: \n',classification_report(y_test,y_pred))
# print('Accuracy is: ',rfc.score(X_test,y_test)) 

sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))
sel_.fit(X_train, np.ravel(y_train,order='C'))
sel_.get_support()
X_train = pd.DataFrame(X_train)

selected_feat = X_train.columns[(sel_.get_support())]
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
np.sum(sel_.estimator_.coef_ == 0)))

removed_feats = X_train.columns[(sel_.estimator_.coef_ == 0).ravel().tolist()]
# print(removed_feats)
X_train_selected = sel_.transform(X_train)
X_test_selected = sel_.transform(X_test)
print(X_train_selected.shape, X_test_selected.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Create a random forest classifier
clf = RandomForestClassifier(random_state = 4)

# clf = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
# Train the classifier
clf.fit(X_train_selected,np.ravel(y_train,order='C'))
# Apply The Full Featured Classifier To The Test Data
y_pred = clf.predict(X_test_selected)
# View The Accuracy Of Our Selected Feature Model
print(accuracy_score(y_test, y_pred))

