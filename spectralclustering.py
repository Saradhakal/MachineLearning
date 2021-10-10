import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


#import Dataset
# df = pd.read_csv("../alzheimer/alzheimer.csv")
df = pd.read_csv("../alzheimer/Copy_alz.csv")


df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})
df['Group'] = df['Group'].map({'Demented': 1, 'Nondemented': 0})
print(df['M/F'])
# replace

column_means = df.mean()
df = df.fillna(column_means)

y = df['Group'].values
X = df[['M/F', 'Age-classification', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']]
# y = df['Group']

# Scaling the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# # # No
# X_normalized =normalize(X_scaled)

# X_normalized = pd.DataFrame(X_normalized)

# Reducing the dimensions of the data


pca = PCA(n_components=2)
X_principal =pca.fit_transform(X_scaled)
# X_principal =pca.fit_transform(X_scaled)

X_principal = pd.DataFrame(X_principal)
X_principal.columns= ['P1','P2']

print(X_principal.head(2))

# Building the clustering model
spectral_model_rbf = SpectralClustering(n_clusters = 2, affinity ='rbf')
# labels = spectral_model_rbf.labels_


# training the model and storing the predicted cluster labels
labels_rbf = spectral_model_rbf.fit_predict(X_principal)

# Visualizing the clustering
plt.scatter(X_principal['P1'], X_principal['P2'],c=SpectralClustering(n_clusters=2,affinity='rbf').fit_predict(X_principal), cmap = plt.cm.winter)
plt.show()

# Building the clustering model 
spectral_model_nn = SpectralClustering(n_clusters = 2, affinity ='nearest_neighbors') 
  
# Training the model and Storing the predicted cluster labels 
labels_nn = spectral_model_nn.fit_predict(X_principal)

# Visualizing the clustering 
plt.scatter(X_principal['P1'], X_principal['P2'], c = SpectralClustering(n_clusters = 2, affinity ='nearest_neighbors') .fit_predict(X_principal), cmap =plt.cm.winter) 
plt.show()

# Performance evaluation
# List of different values of affinity 
affinity = ['rbf', 'nearest-neighbours'] 
  
# List of Silhouette Scores 
s_scores = [] 
  
# Evaluating the performance 
s_scores.append(silhouette_score(df, labels_rbf)) 
s_scores.append(silhouette_score(df, labels_nn)) 
  
# Plotting a Bar Graph to compare the models 
plt.bar(affinity, s_scores) 
plt.xlabel('Affinity') 
plt.ylabel('Silhouette Score') 
plt.title('Comparison of different Clustering Models') 
plt.show() 

print(s_scores)
