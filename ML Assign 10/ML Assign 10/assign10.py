#!/usr/bin/env python
# coding: utf-8

# # Housing Price Prediction Linear Regression
# # Submitted by Harsh Srivastava
# # 117CS0755

# ## importing libraries

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## loading the training dataset

# In[2]:


train_dataset_path = None
for root, dirs, files in os.walk(".", topdown=False) :
    for name in files:
        if name.endswith('train.csv') :
            train_dataset_path = os.path.join(root, name)
            break
    if train_dataset_path != None :
        break
        
train_dataset = pd.read_csv(train_dataset_path)
train_dataset


# ## independent variable and dependent variables

# In[3]:


X = train_dataset.iloc[:, 1:-1]

tot_size = X.shape[0]
print(X.shape)
X


# In[4]:


y = train_dataset['SalePrice'].values
y.shape


# ## removing columns with more than 50 percent NA values

# In[5]:


X_drop_NA = X.dropna(axis = 1, thresh = (0.5 * tot_size))
X_drop_NA


# In[6]:


X_drop_NA.mean()


# ## Replacing NA values

# In[7]:


X_fill = X_drop_NA.fillna(X_drop_NA.mean())
X_fill


# ## removing columns with only a single value in all rows

# In[8]:


for col in X_fill.columns:
    if len(X_fill[col].unique()) == 1:
        X_fill.drop(col,inplace=True,axis=1)
        
X_fill


# ## adding a column of ones

# In[9]:


X_fill = pd.concat([pd.Series(1, index=X_fill.index, name='ones'), X_fill], axis=1)
X_fill


# ## One Hot Encoding the dataframe

# In[10]:


X_one_hot = pd.get_dummies(X_fill)
X_one_hot


# ## normalizing the columns

# In[11]:


X_norm = X_one_hot / X_one_hot.max()
X_norm


# ## splitting train and test data again

# In[12]:


train_size = int(tot_size * 0.8)
test_size = tot_size - train_size

X_train = X_norm.iloc[0: train_size, :]
X_test = X_norm.iloc[train_size: tot_size, :]
y_train = y[0: train_size]
y_test = y[train_size: tot_size]


# In[13]:


X_train


# In[14]:


X_train_np = X_train.to_numpy()
X_train_np


# In[15]:


X_test


# In[16]:


X_test_np = X_test.to_numpy()
X_test_np


# In[17]:


print(y_train)
print(y_test)


# ## Calculating parameters

# In[18]:


B = np.dot(np.dot(np.linalg.pinv(np.dot(X_train_np.T, X_train_np)), X_train_np.T), y_train)
B


# ## predicting for Test dataset

# In[19]:


y_pred = np.dot(X_test, B)
y_pred


# ## RMSE value

# In[20]:


RMSE = np.sqrt(np.sum((y_test - y_pred) ** 2) / test_size)
RMSE


# ## Plotting actual vs predicted values

# In[27]:


plt.figure()
plt.title('Actual Values')
plt.plot(list(range(0, test_size)), y_test, color='green')

plt.figure()
plt.title('Predicted Values')
plt.plot(list(range(0, test_size)), y_pred, label = '', color='red')
plt.show()


# In[ ]:




