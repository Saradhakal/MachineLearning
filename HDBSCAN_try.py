import numpy as np # linear algebra
import pandas as pd 
import sklearn.preprocessing as pp
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, Birch
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix, accuracy_score


from numpy import unique
from sklearn import metrics

import hdbscan

#import Dataset
# df = pd.read_csv("../alzheimer/alzheimer.csv")
df = pd.read_csv("../alzheimer/Copy_alz.csv")


df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})
df['Group'] = df['Group'].map({'Demented': 1, 'Nondemented': 0})
print(df['M/F'])

# replace with mean

column_means = df.mean()
df = df.fillna(column_means)

y = df['Group'].values
X = df[['M/F', 'Age-classification', 'EDUC', 'CDR','SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]
# y = df['Group']

# Scaling the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

  
# Normalizing the Data 

X_normalized =normalize(X_scaled)
X_normalized = pd.DataFrame(X_normalized)

# Reducing the dimensions of the data
pca = PCA(n_components=2)
X_principal =pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns= ['P1','P2']

print(X_principal.head(2))


clusterer = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True)
clusterer.fit(X_principal)

yhat_br = clusterer.fit_predict(X_principal)
clusters_br = unique(yhat_br)
print("Clusters formed",clusters_br)
labels_br = clusterer.labels_
# print(labels_br)


score_br = metrics.silhouette_score(X_principal,labels_br)

print("Score of HDBSCAN = ", score_br)

# # Visualizing the clustering 
# plt.scatter(X_principal['P1'], X_principal['P2'], c = yhat_br) 
# plt.show()

# # Accuracy Score
# acc =  accuracy_score(y, labels_br)

# print("BIRCH clustering\n")
# print("Accuracy Score")
# print(acc)

# X, y = data[:, 1:], data[:,0]

# plt.scatter(normalized_df[:, 1:], normalized_df[:, 0], c = yhat_br)
# plt.show()