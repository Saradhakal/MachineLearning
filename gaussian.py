import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


# df = pd.read_csv("../alzheimer/alzheimer.csv")
df = pd.read_csv("../alzheimer/Copy_alz.csv")



df["SES"].fillna(df["SES"].mean(), inplace=True)
df["MMSE"].fillna(df["MMSE"].mean(), inplace=True)
print(df.isna().sum())
ses = df['SES']
print(ses)

# # Fit a normal distribution to the data:
mu, std = norm.fit(ses)

# # # Plot the histogram.
plt.hist(ses, bins=25, density=True, alpha=0.6, color='g')

# # # Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)

plt.show()


