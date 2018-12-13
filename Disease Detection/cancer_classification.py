#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


from sklearn.datasets import load_breast_cancer


# In[3]:


data = load_breast_cancer()


# In[4]:


data


# In[5]:


data.keys()


# In[6]:


print(data['feature_names'])


# In[7]:


df_data = pd.DataFrame(np.c_[data['data'], data['target']], columns=np.append(data['feature_names'], ['target']))


# In[8]:


df_data.head()


# In[9]:


df_data.tail()


# In[10]:


# Exploratory data analysis

sns.pairplot(df_data, vars=['mean radius', 'mean texture', 'mean area'], hue='target')


# In[11]:


sns.countplot(df_data['target'])


# In[12]:


sns.scatterplot(x='mean area', y='mean smoothness', hue='target', data = df_data)


# In[13]:


# Creating a correlation heatmap

plt.figure(figsize=(20,10))
sns.heatmap(df_data.corr(), annot=True)


# In[ ]:





# In[14]:


# Spit the training sets

from sklearn.model_selection import train_test_split

X = df_data.drop(['target'], axis=1)
Y = df_data['target']


# In[15]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[16]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from time import time

model = SVC()
t = time()
model.fit(X_train, Y_train)
print(f'Time taken = {time() - t} ms')


# In[ ]:




