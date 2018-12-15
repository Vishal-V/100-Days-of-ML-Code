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

sns.pairplot(df_data, vars=['mean radius', 'mean texture', 'mean area', 'worst area'], hue='target')


# In[11]:


sns.countplot(df_data['target'])


# In[12]:


sns.scatterplot(x='mean area', y='mean smoothness', hue='target', data = df_data)


# In[13]:


# Creating a correlation heatmap

plt.figure(figsize=(20,10))
sns.heatmap(df_data.corr(), annot=True)


# In[ ]:





# In[15]:


# Split the training sets

from sklearn.model_selection import train_test_split

X = df_data.drop(['target'], axis=1)
Y = df_data['target']


# In[16]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[17]:


# Normalization

x_min = X_train.min()
range_x = (X_train - x_min).max()
X_train_scaled = (X_train - x_min)/range_x

test_min = X_test.min()
range_test = (X_test - test_min).max()
X_test_scaled = (X_test - test_min)/range_test


# In[18]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from time import time

model = SVC()
t = time()
model.fit(X_train_scaled, Y_train)
print(f'Time taken = {time() - t} ms')


# In[19]:


pred = model.predict(X_test_scaled)
acc = accuracy_score(pred, Y_test)
print(acc)


# In[20]:


cm = confusion_matrix(Y_test, pred)
# plt.figure(figsize=(20,10))
sns.heatmap(cm, annot=True)


# In[21]:


print(classification_report(Y_test, pred))


# In[22]:


from sklearn.model_selection import GridSearchCV

params = {'C':[0.1,1,10,100], 'kernel':['rbf'], 'gamma':[1,0.1,0.01,0.001]}

grid = GridSearchCV(SVC(), params, refit=True, verbose = 4)


# In[23]:


grid.fit(X_train_scaled, Y_train)


# In[24]:


grid.best_params_


# In[25]:


grid_pred = grid.predict(X_test_scaled)
acc = accuracy_score(Y_test, grid_pred)
conf = confusion_matrix(Y_test, grid_pred)
sns.heatmap(conf, annot=True)


# In[26]:


print(classification_report(Y_test, grid_pred))


# In[27]:


model1=SVC(C=1, gamma=1, kernel='rbf')
model1.fit(X_train_scaled, Y_train)
prediction=model1.predict(X_test_scaled)
accuracy = accuracy_score(Y_test, prediction)
print(accuracy)
print(classification_report(Y_test, prediction))


# In[ ]:




