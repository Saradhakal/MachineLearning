import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pandas import read_csv
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


#import Dataset
# df = pd.read_csv("../alzheimer/alzheimer.csv")
df = pd.read_csv("../alzheimer/Copy_alz.csv")

# get the matrix from Dataframe

df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})        # Male is 1, female is 0
df['Group'] = df['Group'].map({'Demented': 1, 'Converted':1, 'Nondemented': 0})     # Demented and converted is 1, Nondemented is 0
df.rename(columns={'M/F': 'Gender'}, inplace = True)

# replace
# df.fillna(0, inplace=True)
# print(df.isna().sum())

# # Replace
# calculate mean
column_means = df.mean()

# Replace
df = df.fillna(column_means)
# print(df.isnull().sum())
data = df.copy() # for VISUALIZATION

# data = df.to_numpy()
# print(data)

#feature Scaling  
from sklearn.preprocessing import StandardScaler    
scaler = StandardScaler()
# transform data
scaled = scaler.fit_transform(data)
X = data.drop("Group",axis=1)
y = data["Group"]
# X1= df.drop("Group",1)
# Separate input and output variables
# X, y = data[:, 1:], data[:,0]


# # Create correlation matrix
# corr_matrix = df.corr()
# print(corr_matrix)

# # Create correlation heatmap
# plt.figure(figsize=(8,6))
# plt.title('Correlation Heatmap of OASIS Dataset')
# a = sns.heatmap(corr_matrix, square=True, annot=True, fmt='.2f', linecolor='black')
# a.set_xticklabels(a.get_xticklabels(), rotation=30)
# a.set_yticklabels(a.get_yticklabels(), rotation=30)           
# plt.show()  

# # Select upper triangle of correlation matrix
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# print(upper)   

# # Find index of feature columns with correlation greater than 0.9
# to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
# print(to_drop)

# new method
data = df.to_numpy()

chi_scores = chi2(X, y)
print(chi_scores)
p_values = pd.Series(chi_scores[0],index = X.columns)
p_values.sort_values(ascending = False , inplace = True)
# print(p_values)
# p_values.plot.bar()

p_values.plot(kind="bar")
plt.xlabel("Features",fontsize=20)
plt.ylabel("Chi-square Score",fontsize=20)
plt.title("Chi-square Test")
plt.show()