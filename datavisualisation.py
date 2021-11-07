

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# import dataframe
df = pd.read_csv("../alzheimer/Copy_alz.csv")

# df = pd.read_csv("../alzheimer/alzheimer.csv")
from sklearn.preprocessing import StandardScaler, normalize

df.head()
print(df.shape)




# sns.countplot(x='Group',data = df)
# plt.show()

# fig, axes = plt.subplots(2,3,figsize = (16,6))
# fig.suptitle("Box Plot",fontsize=14)
# sns.set_style("whitegrid")
# sns.boxplot(data=df['SES'], orient="v",width=0.4, palette="colorblind",ax = axes[0][0])
# sns.boxplot(data=df['EDUC'], orient="v",width=0.4, palette="colorblind",ax = axes[0][1])
# sns.boxplot(data=df['MMSE'], orient="v",width=0.4, palette="colorblind",ax = axes[0][2])
# sns.boxplot(data=df['CDR'], orient="v",width=0.4, palette="colorblind",ax = axes[1][0])
# sns.boxplot(data=df['eTIV'], orient="v",width=0.4, palette="colorblind",ax = axes[1][1])
# sns.boxplot(data=df['ASF'], orient="v",width=0.4, palette="colorblind",ax = axes[1][2])
# plt.show()
# #xlabel("Time");

# cols = ['M/F','Age-classification', 'EDUC', 'SES', 'MMSE', 'CDR','eTIV','nWBV','ASF']
# x=df.fillna('')
# sns_plot = sns.pairplot(x[cols])
# plt.show()

# # y is label and X are features
# y = df['Group'].values
# X = df[['M/F', 'Age-classification', 'EDUC', 'SES', 'MMSE','CDR', 'eTIV', 'nWBV', 'ASF']]
# # y = df['Group']
# df.rename(columns={'M/F': 'Gender'}, inplace = True)

# # print(y)
# # print(X)

# # checking missing values in each column
# print(df.isnull().sum())

cols = df.columns
colours =['g','y']
f,ax = plt.subplots(figsize = (12,8))
sns.set_style("whitegrid")
plt.title("Missing Values")
sns.heatmap(df[cols].isnull(), cmap = sns.color_palette(colours))
plt.show()

# # SES and MMSE have null values , Replace with Mean

# calculate mean
column_means = df.mean()
# print(column_means)

# Replace
df = df.fillna(column_means)
# print(df.isnull().sum())


cols = df.columns
colours =['g','y']
f,ax = plt.subplots(figsize = (12,8))
sns.set_style("whitegrid")
plt.title("After Removing null")
sns.heatmap(df[cols].isnull(), cmap = sns.color_palette(colours))
plt.show()


# # Convert into labels, 
sns.countplot(x='Group',data = df)
plt.show()


df['M/F'] = df['M/F'].map({'M': 1, 'F': 0})        # Male is 1, female is 0
df['Group'] = df['Group'].map({'Demented': 1, 'Converted':1, 'Nondemented': 0})     # Demented and converted is 1, Nondemented is 0


# # Convert into labels, 
sns.countplot(x='Group',data = df)
plt.show()


# # #The lines of code below calculate and print the interquartile range for each of the variables in the dataset.
# # Q1 = df.quantile(0.25)
# # Q3 = df.quantile(0.75)
# # IQR = Q3 - Q1
# # print("The IQR is:\n",IQR)

# # print("The above output prints the IQR scores, which can be used to detect outliers") 
# # print("The code below generates an output with the 'True' and 'False' values")
# # print("Points where the values are 'True' represent the presence of the outlier")
# # print(df < (Q1 - 1.5 * IQR)) or (df > (Q3 + 1.5 * IQR))

# # Histogram

# df.hist(column='M/F', bins= 10)
# plt.ylabel('frequency')
# plt.show()

# df.hist(column='Age-classification', bins= 10)
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


# # Heatmap
# corr = df.corr()
# plt.figure(figsize=(12,6))
# sns.heatmap(corr, annot=True, vmin=-1)
# plt.show()

# Relaionship between gender and dementia
# demented_group = df[df['Group']==1]['Gender'].value_counts()
# demented_group = pd.DataFrame(demented_group)
# demented_group.index=['Male', 'Female']
# demented_group.plot(kind='bar', figsize=(8,6))
# plt.title('Gender vs Dementia', size=16)
# plt.xlabel('Gender', size=14)
# plt.ylabel('Patients with Dementia', size=14)
# plt.xticks(rotation=0)
# plt.show()

# # Relaionship between gender and dementia
# demented_group = df[df['Group']==0]['Gender'].value_counts()
# demented_group = pd.DataFrame(demented_group)
# demented_group.index=['Male', 'Female']
# demented_group.plot(kind='bar', figsize=(8,6))
# plt.title('Gender vs No-Dementia', size=16)
# plt.xlabel('Gender', size=14)
# plt.ylabel('Patients with No-Dementia', size=14)
# plt.xticks(rotation=0)
# plt.show()



# # Scatter Plot

# plt.scatter(x= df['ASF'] , y= df['eTIV'])
# plt.xlabel('asf') #x label
# plt.ylabel('educ') #y label
# plt.show()



# # # Scale the dataset
# # # here StandardScaler() means z = (x - u) / s
# # scaler = StandardScaler().fit(X)
# # #scaler = MinMaxScaler().fit(X_trainval)
# # X_scaled = scaler.transform(X)
# # print(X_scaled)





