#pandas

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns

#import Dataset
df = pd.read_csv("../alzheimer/alzheimer.csv")

# df = pd.read_csv("../alzheimer/alzheimer.csv")

df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})
df['Group'] = df['Group'].map({'Demented': 1, 'Nondemented': 0})
# print(df['M/F'])

y = df['Group']

# replace

column_means = df.mean()
df = df.fillna(column_means)

# Preprocessing the data to make it visualizable
  
# Scaling the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
  
# Normalizing the Data
X_normalized = normalize(X_scaled)
  
# Converting the numpy array into a pandas DataFrame
X_normalized = pd.DataFrame(X_normalized)
  
# Reducing the dimensions of the data
pca = PCA(n_components = 2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']
  
X_principal.head()
print(X_principal)



# print((df.replace("NaN", "0",inplace=True)))

# print(list(df.columns))
# print(df.describe())

# df.hist(column='Age', bins= 10)
# plt.ylabel('frequency')
# plt.show()

# df.hist(column='EDUC', bins= 10)
# plt.ylabel('frequency')
# plt.show()

# df.hist(column='SES', bins= 10)
# plt.ylabel('frequency')
# plt.show()

# df.hist(column='MMSE', bins= 10)
# plt.ylabel('frequency')
# plt.show()

# df.hist(column='CDR', bins= 10)
# plt.ylabel('frequency')
# plt.show()

# df.hist(column='eTIV', bins= 10)
# plt.ylabel('frequency')
# plt.show()

# df.hist(column='nWBV', bins= 10)
# plt.ylabel('frequency')
# plt.show()

# df.hist(column='ASF', bins= 10)
# plt.ylabel('frequency')
# plt.show()

# box-plot for 'Age'
# sns.boxplot(y=df["EDUC"])
# plt.show()
# plt.xticks(rotation=90)
# plt.ylabel('EDUC')

# box-plot for 'Age'
# sns.boxplot(y=df["EDUC"])
# plt.show()
# plt.xticks(rotation=90)
# plt.ylabel('EDUC')

# box-plot for 'Age'
# sns.boxplot(y=df["EDUC"])
# plt.show()
# plt.xticks(rotation=90)
# plt.ylabel('EDUC')

# box-plot for 'Age'
# sns.boxplot(y=df["EDUC"])
# plt.show()
# plt.xticks(rotation=90)
# plt.ylabel('EDUC')


# # Scatter Plot
# x= df['eTIV']
# y= df['ASF']
# plt.scatter(x, y)
# plt.xlabel('eTIV') #x label
# plt.ylabel('ASF') #y label
# plt.show()

# Scatter Plot
# y= df['eTIV']
# x= df['ASF']
# plt.scatter(x, y)
# plt.ylabel('eTIV') #x label
# plt.xlabel('ASF') #y label
# plt.show()




# Country = ['USA','Canada','Germany','UK','France']
# GDP_Per_Capita = [45000,42000,52000,49000,47000]
# count = [1,2,3]

# figure = plt.figure(figsize=(12,6))

# cormat = df.corr()
# cc= round(cormat,3)
# print(cc)

# sns.heatmap(cormat,data = df,annot= True,cmap = 'Reds')
# plt.show()
# # df.info()


