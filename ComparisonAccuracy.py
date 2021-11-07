
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm


from pandas import read_csv
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


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

# # # separate the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 1)

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()  
x_train= st_x.fit_transform(X_train)    
x_test= st_x.transform(X_test)    

# Adaboost
adaboost=AdaBoostClassifier()
adaboost.fit(x_train, y_train)
y_pred=adaboost.predict(x_test)

#Fitting Decision Tree classifier to the training set  
dt_classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)  
dt_classifier.fit(x_train, y_train)  
y_pred= dt_classifier.predict(x_test)  

# ExtraTreeClassifier
et_model = ExtraTreeClassifier()
et_model.fit(x_train, y_train)
y_pred = et_model.predict(x_test)

# GradientBoost Classifier
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)
y_pred = gb_model.predict(X_test)

# K Neighbours
knn_model=KNeighborsClassifier() 
knn_model.fit(x_train,y_train)
y_pred=knn_model.predict(X_test)

# Logistic Regression
LogReg_clf = LogisticRegression(random_state = None, max_iter=100)
LogReg_clf.fit(x_train, y_train)
y_pred = LogReg_clf.predict(x_test)

# Naive Bayes

gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)

# Random Forest 
rf_classifier = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5, n_estimators=100, oob_score=True)
rf_classifier.fit(x_train, y_train)
y_pred = rf_classifier.predict(x_test)

# Suppot Vector Machine
svclassifier = SVC(kernel = 'linear')
svclassifier.fit(x_train, y_train)
y_pred = svclassifier.predict(x_test)

# models = [adaboost,dt_classifier,kntuned,mlpc,cartctuned,rfctuned,gbmc,xgbc,lgbmctuned,catbctuned]
models = [adaboost,dt_classifier,et_model,gb_model,knn_model,LogReg_clf,gnb,rf_classifier,svclassifier]

r = pd.DataFrame(columns=["MODELS","ACC"])

for model in models:
    name = model.__class__.__name__
    predict = model.predict(x_test)
    accuracy = accuracy_score(y_test, predict)

    print("-" * 28)
    print(name + ": ")
    print(f"Accuracy: {accuracy}")
    result = pd.DataFrame([[name,accuracy*100]],columns=["MODELS","ACC"])
    r = r.append(result)
    
sns.barplot(x="ACC",y="MODELS",data=r,color="g")
plt.xlabel("ACC")
plt.title("MODEL ACCURACY COMPARISON")
plt.show()

r = pd.DataFrame(columns=["MODELS","PRECISION"])

for model in models:
    name = model.__class__.__name__
    predict = model.predict(x_test)
    precision = precision_score(y_test,predict)

    print("-" * 28)
    print(name + ": ")
    print(f"Precision: {precision}")
    result = pd.DataFrame([[name,precision*100]],columns=["MODELS","PRECISION"])
    r = r.append(result)
    
sns.barplot(x="PRECISION",y="MODELS",data=r,color="b")
plt.xlabel("PRE")
plt.title("MODEL PRECISION COMPARISON")
plt.show()