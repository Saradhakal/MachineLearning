
import numpy as np # linear algebra
import pandas as pd 
import sklearn.preprocessing as pp
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, Birch, KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score


from numpy import unique
from sklearn import metrics


#import Dataset
df = pd.read_csv("../alzheimer/alzheimer.csv")

df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})
df['Group'] = df['Group'].map({'Demented': 1, 'Nondemented': 0})
print(df['M/F'])

y = df['Group']

# replace

column_means = df.mean()
df = df.fillna(column_means)

# Scaling the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

#K means Clustering 
# def doKmeans(X, nclust=4):
#     model = KMeans(nclust)
#     model.fit(X)
#     clust_labels = model.predict(X)
#     cent = model.cluster_centers_
#     return (clust_labels, cent)

# clust_labels, cent = doKmeans(X_scaled, 2)
# kmeans = pd.DataFrame(clust_labels)
# df.insert((df.shape[1]),'kmeans',kmeans)

kmeans = KMeans(n_clusters=2, random_state = 0)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# check how many of the samples were correctly labeled

correct_labels = sum(y == labels)

print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))

print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))

# print("Overall K-Means accuracy %2f%%" % (accuracy_score(class_labels, clusters)* 100))



#Plot the clusters obtained using k means
# fig = plt.figure()
# ax = fig.add_subplot(111)
# scatter = ax.scatter(df['Age'],df['MMSE'],s=50)
# ax.set_title('K-Means Clustering')
# ax.set_xlabel('AGE')
# ax.set_ylabel('MMSE')
# plt.colorbar(scatter)
# plt.show()

# # Standardize data
# scaler = pp.StandardScaler() 
# scaled_df = scaler.fit_transform(df) 
  
# # Normalizing the Data 

# X_normalized =normalize(X_scaled)

# X_normalized = pd.DataFrame(X_normalized)

# # Reducing the dimensions of the data
# pca = PCA(n_components=2)
# X_principal =pca.fit_transform(X_normalized)
# X_principal = pd.DataFrame(X_principal)
# X_principal.columns= ['P1','P2']

# print(X_principal.head(2))



# model_br = Birch(threshold=0.01, n_clusters=5)
# model_br.fit(X_principal)
# #
# yhat_br = model_br.predict(X_principal)
# clusters_br = unique(yhat_br)
# print("Clusters of Birch",clusters_br)
# labels_br = model_br.labels_

# score_br = metrics.silhouette_score(X_principal,labels_br)

# print("Score of Birch = ", score_br)

# # Visualizing the clustering 
# plt.scatter(X_principal['P1'], X_principal['P2'], c = yhat_br) 
# plt.show()

