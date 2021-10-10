import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

# df = pd.read_csv("../alzheimer/alzheimer.csv")
df = pd.read_csv("../alzheimer/Copy_alz.csv")


df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})
df['Group'] = df['Group'].map({'Demented': 1, 'Nondemented': 0})
# print(df['M/F'])

df["SES"].fillna(df["SES"].median(), inplace=True)
df["MMSE"].fillna(df["MMSE"].mean(), inplace=True)
# print(df.isna().sum())


#AGE 
plt.hist(df['Age-classification'])
plt.show()


# Standard deviation and mean
std = np.std(df['Age-classification'],ddof=1)
mean = np.mean(df['Age-classification'])

# plotting
domain = np.linspace(np.min(df['Age-classification']),np.max(df['Age-classification']))
plt.plot(domain, norm.pdf(domain, mean, std),
label = '$\mathcal{N}$' + f'$(\mu \\approx{round(mean)}, \sigma\\approx{round(std)} )$')
plt.hist(df['Age-classification'], edgecolor='black', alpha = .5, density = True )
plt.title('Age')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()



#SES 
plt.hist(df['SES'])
plt.show()


# Standard deviation and mean
std = np.std(df['SES'],ddof=1)
mean = np.mean(df['SES'])

# plotting
domain = np.linspace(np.min(df['SES']),np.max(df['SES']))
plt.plot(domain, norm.pdf(domain, mean, std),
label = '$\mathcal{N}$' + f'$(\mu \\approx{round(mean)}, \sigma\\approx{round(std)} )$')
plt.hist(df['SES'], edgecolor='black', alpha = .5, density = True )
plt.title('SES')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()



       
