# evaluate model performance with outliers removed using local outlier factor
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_absolute_error

#import Dataset
df = read_csv("../alzheimer/alzheimer.csv")

# retrieve DataFrame's content as a matrix
data = df.values

# split into input and output elements
X, y = data[:, 1:], data[:, -1]

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# summarize the shape of the training dataset
print(X_train.shape, y_train.shape)

# identify outliers in the training dataset
lof = LocalOutlierFactor()
y_hat = lof.fit_predict(X_train)

# select all rows that are not outliers
mask = y_hat != -1
X_train, y_train = X_train[mask, :], y_train[mask]

