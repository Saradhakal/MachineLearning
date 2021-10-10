import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.decomposition import PCA
from sklearn import datasets

df = pd.read_csv("../alzheimer/alzheimer.csv")

df.head()

# get the matrix from Dataframe
df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})
df['Group'] = df['Group'].map({'Demented': 1, 'Nondemented': 0})

# replace
df.fillna(0, inplace=True)
print(df.isna().sum())

column_means = df.mean()
df = df.fillna(column_means)
# data = df.to_numpy()


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# data_rescaled = scaler.fit_transform(df)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
StandardScaler()

data_rescaled = scaler.transform(df)

pca = PCA().fit(data_rescaled)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

# 
# pca = PCA(n_components = 2)
# pca.fit(data_rescaled)
# PCA(n_components=2)

# x_pca = pca.transform(data_rescaled)
# print(data_rescaled)
# print(x_pca.shape)

# plt.scatter(x_pca[:,0],x_pca[:,1], cmap='rainbow')
# plt.xlabel('First principal component')
# plt.ylabel('2nd principal component')
# plt.show()

