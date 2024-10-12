#!/usr/bin/env python
# coding: utf-8

# 1+1+3
# #the answer is 5

# In[3]:


a=5
b=7
c=a*b
c


# In[1]:


import pandas as pd
dataset=("The data set.csv")
read=pd.read_csv(dataset)
read


# In[3]:


#I have succesfully uplaoded my dataset


# In[5]:


read.info()


# In[7]:


read.head()


# In[21]:


read.tail()


# In[11]:


read.duplicated().sum()


# In[19]:


read.describe()


# In[23]:


read.isnull().sum()


# In[49]:


# Box Plot
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8,8))
sns.boxplot(x='HouseNetWorth', y='HousePrice', data=read)
plt.title('House Price across House Net Worth Categories')
plt.show() 


# In[61]:


# Line chart for HousePrice over StoreArea
plt.figure(figsize=(4, 4))
sns.lineplot(x='StoreArea', y='HousePrice', data=read, color='blue')
plt.title('House Price vs Store Area')
plt.xlabel('Store Area')
plt.ylabel('House Price')
plt.show()


# In[73]:


# Line chart for HousePrice over BasementArea
plt.figure(figsize=(4, 4))
sns.lineplot(x='BasementArea', y='HousePrice', data=read, color='red')
plt.title('House Price vs Basement Area')


# In[75]:


# Line chart for HousePrice over LawnArea
plt.figure(figsize=(4, 4))
sns.lineplot(x='LawnArea', y='HousePrice', data=read, color='green')
plt.title('House Price vs Lawn Area')


# In[77]:


# Line chart for HousePrice over HouseNetWorth
plt.figure(figsize=(4, 4))
sns.lineplot(x='HouseNetWorth', y='HousePrice', data=read, color='orange')
plt.title('House Price vs House Net Worth')


# In[79]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set style
sns.set_style("whitegrid")

# pairplot with hue
plt.figure(figsize=(10, 8))
sns.pairplot(read, hue="HouseNetWorth", palette="husl")
plt.title('Pairplot with Different Colors based on House Net Worth')
plt.show()


# In[7]:


#Remove Outliers

import pandas as pd

#the dataframe
dataset=("The data set.csv")
read=pd.read_csv(dataset)

#Identify numerical columns
numeric_read=read.select_dtypes(include='number')

#Calculate Q1 and Q3 and IQR
Q1=numeric_read.quantile(0.25)
Q3=numeric_read.quantile(0.75)
IQR=Q3-Q1

#Define outlier bounds
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR

#Create a mask for outliers
outlier_mask=~((numeric_read<lower_bound)|(numeric_read>upper_bound)).any(axis=1)

#Filter the Dataframe to remove outliers,keeping non-numerical columns
read_no_outliers=read[outlier_mask]

#Display the results
print("Original Dataframe:")
print(read)
print("\nDataFrame without outliers:")
print(read_no_outliers)


# In[17]:


#Generating a heatmap

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#load the dataset
read=pd.read_csv('The data set.csv')

#Display the first rew rows of the dataset
print("Original Dataframe:")
print(read.head())

#Identify numeric columns
numeric_read=read.select_dtypes(include='number')

#Caalculate the correlation matrix
correlation_matrix=numeric_read.corr()

#Generate the heatmap
plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',
           square=True,linewidths=.5,cbar_kws={"shrink":.8})

#Add Title
plt.title('Correlation Heatmap')

#show the plot
plt.show()


# In[ ]:




