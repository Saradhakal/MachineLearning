import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
import seaborn as sns

from sklearn.metrics import precision_recall_fscore_support as score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn import metrics #accuracy measure

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

# Convert into labels 
df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})        # Male is 1, female is 0
df['Group'] = df['Group'].map({'Demented': 1, 'Converted':1, 'Nondemented': 0})     # Demented and converted is 1, Nondemented is 0
df.rename(columns={'M/F': 'Gender'}, inplace = True)

# # y is label and X are features
# X = df[['Gender', 'Age-classification', 'EDUC', 'SES', 'MMSE','CDR', 'eTIV', 'nWBV', 'ASF']]
# y = df['Group'].values
# # y = df['Group']
# # 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel

numerics = ['int16','int32','int64','float16','float32','float64']
numerical_vars = list(df.select_dtypes(include=numerics).columns)
data = df[numerical_vars]
# print(data.shape)

data = df.to_numpy()
# print(data)

# X1= df.drop("Group",1)
# Separate input and output variables
X, y = data[:, 1:], data[:,0]
print(X.shape)
# from sklearn.preprocessing import MinMaxScaler
# Min_Max = MinMaxScaler()
# X = Min_Max.fit_transform(X)

from sklearn.preprocessing import StandardScaler

# Scaling
scaler = StandardScaler().fit(X)
X_scale = scaler.transform(X)
# Y_scale = scaler.transform(y)


sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))
sel_.fit(X_scale, np.ravel(y,order='C'))
# print(sel_.get_support())s
X_scale = pd.DataFrame(X_scale)
selected_feat = X_scale.columns[(sel_.get_support())]
print('total features: {}'.format((X_scale.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(np.sum(sel_.estimator_.coef_ == 0)))


removed_feats = X_scale.columns[(sel_.estimator_.coef_ == 0).ravel().tolist()]
print('R F',removed_feats)



X_selected = sel_.transform(X_scale)
# X_test_selected = sel_.transform(X)
# print(X_selected.shape)
# print(y.shape)
# print(X_train_selected.shape, X_test_selected.shape)

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.33, random_state = 1)

from sklearn.ensemble import AdaBoostClassifier
Model=AdaBoostClassifier()
Model.fit(X_train, y_train)
y_pred=Model.predict(X_test)

# Summary of the predictions made by the classifier

# Accuracy score
acc = accuracy_score(y_pred,y_test)

# Precicion score
# When it predicts yes, how often is it correct?
precision = precision_score(y_test,y_pred)

# Recall/sensitivity
# When it’s actually yes, how often does it predict yes?
recall = recall_score(y_test,y_pred)

# f1 score (Average of weighted score)
f1 = f1_score(y_test,y_pred)

# recall_sensitivity = metrics.recall_score(y_test, y_pred, pos_label=1)
recall_specificity = (metrics.recall_score(y_test, y_pred, pos_label=0))

print("\nAda Boost\n")
print("Accuracy Score", acc*100)
print("Precision Score",precision*100)
print("Recall score",recall*100)
print("f1_score",f1*100)
print('FPR', (1-recall_specificity)*100)


from sklearn import tree 
from sklearn.tree import DecisionTreeClassifier

classifier= DecisionTreeClassifier(criterion='entropy', random_state=1)  
# classifier = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  


classifier.fit(X_train, y_train)  

#Predicting the test set result  
y_pred1= classifier.predict(X_test)  

# Accuracy Score 
# How often is the classifier correct?
# acc =  classifier.score(x_test,y_test)
acc = accuracy_score(y_pred1,y_test)

# Precicion score
# When it predicts yes, how often is it correct?
precision = precision_score(y_test,y_pred1)

# Recall/sensitivity
# When it’s actually yes, how often does it predict yes?
recall = recall_score(y_test,y_pred1)

# f1 score
f1 = f1_score(y_test,y_pred1)


print("Decision Tree\n")
# recall_sensitivity = metrics.recall_score(y_test, y_pred, pos_label=1)
recall_specificity = (metrics.recall_score(y_test, y_pred1, pos_label=0))

print("Accuracy Score", acc*100)
print("Precision Score",precision*100)
print("Recall score",recall*100)
print("f1_score",f1*100)
print('FPR', (1-recall_specificity)*100)


# ExtraTreeClassifier
from sklearn.tree import ExtraTreeClassifier

et_classifier = ExtraTreeClassifier()

et_classifier.fit(X_train, y_train)

y_pred2 = et_classifier.predict(X_test)

# Summary of the predictions made by the classifier

# Accuracy score
acc = accuracy_score(y_pred2,y_test)

# Precicion score
# When it predicts yes, how often is it correct?
precision = precision_score(y_test,y_pred2)

# Recall/sensitivity
# When it’s actually yes, how often does it predict yes?
recall = recall_score(y_test,y_pred2)

# f1 score
f1 = f1_score(y_test,y_pred2)

print("Extra Tree\n")
# recall_sensitivity = metrics.recall_score(y_test, y_pred, pos_label=1)
recall_specificity = (metrics.recall_score(y_test, y_pred2, pos_label=0))

print("Accuracy Score", acc*100)
print("Precision Score",precision*100)
print("Recall score",recall*100)
print("f1_score",f1*100)
print('FPR', (1-recall_specificity)*100)


from sklearn.ensemble import GradientBoostingClassifier
gb_classifier= GradientBoostingClassifier()
gb_classifier.fit(X_train, y_train)
y_pred3 = gb_classifier.predict(X_test)

# Summary of the predictions made by the classifier

# Accuracy score
acc = accuracy_score(y_pred3,y_test)

# Precicion score
# When it predicts yes, how often is it correct?
precision = precision_score(y_test,y_pred3)

# Recall/sensitivity
# When it’s actually yes, how often does it predict yes?
recall = recall_score(y_test,y_pred3)

# f1 score
f1 = f1_score(y_test,y_pred3)

print("\nGradient Boost\n")
# recall_sensitivity = metrics.recall_score(y_test, y_pred, pos_label=1)
recall_specificity = (metrics.recall_score(y_test, y_pred3, pos_label=0))

print("Accuracy Score", acc*100)
print("Precision Score",precision*100)
print("Recall score",recall*100)
print("f1_score",f1*100)
print('FPR', (1-recall_specificity)*100)


# model
from sklearn.neighbors import KNeighborsClassifier #KNN

knn_classifier=KNeighborsClassifier(n_neighbors=5) 

# model.fit(X_train,y_train)
knn_classifier.fit(X_train,y_train)

y_pred4 =knn_classifier.predict(X_test)
print('The accuracy of the KNN is',accuracy_score(y_pred4,y_test))

# Precicion score
# When it predicts yes, how often is it correct?
precision = precision_score(y_test,y_pred4)

# Recall/sensitivity
# When it’s actually yes, how often does it predict yes?
recall = recall_score(y_test,y_pred4)

# f1 score
f1 = f1_score(y_test,y_pred4)

print("\KNN\n")
# recall_sensitivity = metrics.recall_score(y_test, y_pred, pos_label=1)
recall_specificity = (metrics.recall_score(y_test, y_pred4, pos_label=0))

print("Precision Score",precision*100)
print("Recall score",recall*100)
print("f1_score",f1*100)
print('FPR', (1-recall_specificity)*100)


from sklearn.linear_model import LogisticRegression

LogReg_clf = LogisticRegression(random_state = 0)

LogReg_clf.fit(X_train,y_train)
y_pred5=LogReg_clf.predict(X_test)

# Summary of the predictions made by the classifier

# Accuracy score
acc = accuracy_score(y_pred5,y_test)

# Precicion score
# When it predicts yes, how often is it correct?
precision = precision_score(y_test,y_pred5)

# Recall/sensitivity
# When it’s actually yes, how often does it predict yes?
recall = recall_score(y_test,y_pred5)

# f1 score
f1 = f1_score(y_test,y_pred5)

print("\n Logistic Regression\n")
# recall_sensitivity = metrics.recall_score(y_test, y_pred, pos_label=1)
recall_specificity = (metrics.recall_score(y_test, y_pred5, pos_label=0))

print("Accuracy Score", acc*100)
print("Precision Score",precision*100)
print("Recall score",recall*100)
print("f1_score",f1*100)
print('FPR', (1-recall_specificity)*100)


from sklearn.naive_bayes import GaussianNB
# training the model on training set
gnb = GaussianNB()
gnb.fit(X_train, y_train)
  
# making predictions on the testing set
y_pred6 = gnb.predict(X_test)
  
# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
acc = metrics.accuracy_score(y_test, y_pred6)

# Precicion score
# When it predicts yes, how often is it correct?
precision = precision_score(y_test,y_pred6)

# Recall/sensitivity
# When it’s actually yes, how often does it predict yes?
recall = recall_score(y_test,y_pred6)

# f1 score
f1 = f1_score(y_test,y_pred6)

print("\n Naive Bayes\n")
# recall_sensitivity = metrics.recall_score(y_test, y_pred, pos_label=1)
recall_specificity = (metrics.recall_score(y_test, y_pred6, pos_label=0))

print("Accuracy Score", acc*100)
print("Precision Score",precision*100)
print("Recall score",recall*100)
print("f1_score",f1*100)
print('FPR', (1-recall_specificity)*100)


from sklearn.ensemble import RandomForestClassifier
# training the model on training set
rf_classifier = RandomForestClassifier(random_state=1, n_jobs=-1, max_depth=5, n_estimators=100, oob_score=True)
rf_classifier.fit(X_train, y_train)

# # evaluate the model
y_pred7 = rf_classifier.predict(X_test)
# print(y_pred)

# Accuracy Score 
# How often is the classifier correct?
# acc =  rf_classifier.score(X_test,y_test)
# acc = accuracy_score(y_pred7,y_test)
acc=metrics.accuracy_score(y_test, y_pred7)


# Precicion score
# When it predicts yes, how often is it correct?
precision = precision_score(y_test,y_pred7)

# Recall/sensitivity
# When it’s actually yes, how often does it predict yes?
recall = recall_score(y_test,y_pred7)

# f1 score
f1 = f1_score(y_test,y_pred7)

print("\nRandom Forest Classifier\n")
# recall_sensitivity = metrics.recall_score(y_test, y_pred, pos_label=1)
recall_specificity = (metrics.recall_score(y_test, y_pred7, pos_label=0))

print("Accuracy Score", acc*100)
print("Precision Score",precision*100)
print("Recall score",recall*100)
print("f1_score",f1*100)
print('FPR', (1-recall_specificity)*100)


from sklearn import svm
from sklearn.svm import SVC
svclassifier = svm.SVC(kernel = 'linear')
svclassifier.fit(X_train, y_train)

# predict
y_pred8 = svclassifier.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation

# Model Accuracy: how often is the classifier correct?
acc=metrics.accuracy_score(y_test, y_pred8)

# Precicion score
# When it predicts yes, how often is it correct?
precision = precision_score(y_test,y_pred8)

# Recall/sensitivity
# When it’s actually yes, how often does it predict yes?
recall = recall_score(y_test,y_pred8)

# f1 score
f1 = f1_score(y_test,y_pred8)

# Evaluate
print("Support Vector Machine\n")
# recall_sensitivity = metrics.recall_score(y_test, y_pred, pos_label=1)
recall_specificity = (metrics.recall_score(y_test, y_pred8, pos_label=0))

print("Accuracy Score", acc*100)
print("Precision Score",precision*100)
print("Recall score",recall*100)
print("f1_score",f1*100)
print('FPR', (1-recall_specificity)*100)
