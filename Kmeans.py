import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.datasets as dt

from scipy.stats import mode
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# a function that maps K-Means Cluster labels to actual class labels
def map_cluster_to_group(clus, grp, k):
        labels = np.zeros_like(clus)
        for i in range(k):
                mask = (clus == i)
                labels[mask] = mode(grp[mask])[0]
        return labels


#import Dataset
# df = pd.read_csv("../alzheimer/alzheimer.csv")
df = pd.read_csv("../alzheimer/Copy_alz.csv")


df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})
df['Group'] = df['Group'].map({'Demented': 1, 'Nondemented': 0})
print(df['M/F'])

# replace
df.fillna(0, inplace=True)
print(df.isna().sum())

column_means = df.mean()
df = df.fillna(column_means)


# two clusters
km2 = KMeans(n_clusters=2).fit(df)

df['Labels'] = km2.labels_
plt.figure(figsize=(12, 8))
sns.scatterplot(df['Age-classification'], df['CDR'], hue=df['Labels'], 
                palette=sns.color_palette('hls', 2))
plt.title('KMeans with 2 Clusters')
plt.show()

# three clusters
# km2 = KMeans(n_clusters=2).fit(df)

# df['Labels'] = km2.labels_
# plt.figure(figsize=(12, 8))
# sns.scatterplot(df['Age'], df['MMSE'], hue=df['Labels'], 
#                 palette=sns.color_palette('hls', 2))
# plt.title('KMeans with 2 Clusters')
# plt.show()