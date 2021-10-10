# The value of K at which improvement in distortion declines the most is called the elbow, 
# at which we should stop dividing the data into further clusters.


# Import required packages
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



#import Dataset
df = pd.read_csv("../alzheimer/alzheimer.csv")

# get the matrix from Dataframe
print(df['M/F'])
print(df['Group'])
df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})
df['Group'] = df['Group'].map({'Demented': 1, 'Nondemented': 0})
print(df['M/F'])

# replace
df.fillna(0, inplace=True)
print(df.isna().sum())

column_means = df.mean()
df = df.fillna(column_means)

mms = MinMaxScaler()
mms.fit(df)

data_transformed = mms.transform(df)

Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


