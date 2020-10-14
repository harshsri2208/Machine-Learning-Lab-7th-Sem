#!/usr/bin/env python
# coding: utf-8

# # Comparing Distance Based Classifiers

# ## Submitted By Harsh Srivastava
# ## 117CS0755

# ### importing libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


# ### function to calculate euclidean distance between two vectors

# In[2]:


def euclidean(a, b) :
    dist = a - b
    sq_dist = np.dot(np.transpose(dist), dist)
    sq_dist = np.sqrt(sq_dist)
    return sq_dist


# ### function to calculate city block distance between two vectors

# In[3]:


def city_block(a, b) :
    dist = np.abs(a - b)
    return np.sum(dist)


# ### function to calculate chess board distance between two vectors

# In[4]:


def chess_board(a, b) :
    dist = np.abs(a - b)
    return max(dist)


# ### function to calculate cosine distance between two vectors

# In[5]:


def cos_dist(a, b) :
    dot_product = np.dot(a, b.T)
    norm_a = np.linalg.norm(a) 
    norm_b = np.linalg.norm(b) 
    return 1 - (dot_product / (norm_a * norm_b))


# ### function to calculate bray curtis distance between two vectors

# In[6]:


def bray_curtis(a, b) :
    d1 = np.sum(np.abs(a - b))
    d2 = np.sum(np.abs(a + b))
    return d1 / d2


# ### function to calculate canberra distance between two vectors

# In[7]:


def canberra(a, b) :
    dist = np.abs(a - b) / (np.abs(a) + np.abs(b))
    return np.sum(dist)


# ### function to calculate mahalonobis distance between two vectors

# In[8]:


def mahalonobis(a, b, input_space) :
    cov = np.cov(input_space.T)
    cov_inv = np.linalg.inv(cov)
    diff = a - b
    dist = np.dot(np.dot(diff.T, cov_inv), diff)
    return dist


# ### function to calculate correlation distance between two vectors

# In[9]:


def correlation(a, b) :
    dev_a = a - np.mean(a)
    dev_b = b - np.mean(b)
    norm_a = np.linalg.norm(dev_a) 
    norm_b = np.linalg.norm(dev_b) 
    dist = 1 - (np.dot(dev_a, dev_b.T) / (norm_a * norm_b))
    return dist


# ### function to calculate minkowski distance between two vectors

# In[10]:


def minkowski(a, b, p) :
    diff = np.abs(a - b)
    dist = pow(np.sum(pow(diff, p)), (1 / p))
    return dist


# ### fucntion to select a distance type

# In[11]:


def distance(a, b, input_space, dist_type = 0) :
    
    if dist_type == 0 :
        return euclidean(a, b)
    
    elif dist_type == 1 :
        return city_block(a, b)
    
    elif dist_type == 2 :
        return chess_board(a, b)
    
    elif dist_type == 3 :
        return cos_dist(a, b)
    
    elif dist_type == 4 :
        return bray_curtis(a, b)
    
    elif dist_type == 5:
        return canberra(a, b)
    
    elif dist_type == 6:
        return mahalonobis(a, b, input_space)
    
    elif dist_type == 7:
        return correlation(a, b)
    
    elif dist_type == 8:
        return minkowski(a, b, np.random.randint(1, 10))


# ### distance types dictionary

# In[12]:


dist_dict = {0: 'Euclidean',
                1: 'City Block',
                2: 'Chess Board',
                3: 'Cosine',
                4: 'Bray Curtis',
                5: 'Canberra',
                6: 'Mahalonobis',
                7: 'Correlation',
                8: 'Minkowski'}

dist_nums = len(dist_dict)


# ### splitting dataset into 3 classes

# In[13]:


dataset = pd.read_csv('IRIS.csv')

dataset_class1 = dataset.loc[0:49, :]
dataset_class2 = dataset.loc[50:99, :]
dataset_class3 = dataset.loc[100:149, :]


# ### Iris-setosa

# In[14]:


dataset_class1


# ### Iris-versicolor

# In[15]:


dataset_class2


# ### Iris-virginica

# In[16]:


dataset_class3


# ### creating a dictionary to map class name with our defined labels
# ### 'Iris-setosa' --> 0
# ### 'Iris-versicolor' --> 1
# ### 'Iris-virginica' --> 2

# In[17]:


iris_dict = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
iris_dict


# ### splitting features and classes

# In[18]:


features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

X1 = dataset_class1[features]
X2 = dataset_class2[features]
X3 = dataset_class3[features]

X1 = normalize(X1, 'l2')
X2 = normalize(X2, 'l2')
X3 = normalize(X3, 'l2')


# ### splitting dataset into training and test

# In[19]:


from sklearn.model_selection import train_test_split

x_train1_pd, x_test1_pd = train_test_split(X1, test_size=.4)
x_train2_pd, x_test2_pd = train_test_split(X2, test_size=.4)
x_train3_pd, x_test3_pd = train_test_split(X3, test_size=.4)

num_test = 20
num_train = 50 - num_test
total_num_test = 60
total_num_train = 150 - total_num_test

x_train1_pd


# ### converting training and test datasets to numpy arrays for easy calculations

# In[20]:


x_train = []
x_test = []

x_train.append(x_train1_pd)#.to_numpy())
x_train.append(x_train2_pd)#.to_numpy())
x_train.append(x_train3_pd)#.to_numpy())

x_test.append(x_test1_pd)#.to_numpy())
x_test.append(x_test2_pd)#.to_numpy())
x_test.append(x_test3_pd)#.to_numpy())

print(x_train[0])
print(x_test[0])

print(x_train[0].shape)
print(x_test[0].shape)


# ### calculating mean of features of classes in training dataset

# In[21]:


mean = []

mean.append(np.mean(x_train[0], axis = 0))
mean.append(np.mean(x_train[1], axis = 0))
mean.append(np.mean(x_train[2], axis = 0))

print(mean[0])
print(mean[1])
print(mean[2])


# ### predicting classes for iris-setosa test data using euclidean distance

# In[22]:


def prediction_distance_types(dist_type = 0) :

    correct_count = 0 # variable to count number of training examples correctly classified
    pred = []
    error = []

    for i in range(3) :
        train = x_train[i] # for covariance matrix
        test = x_test[i]

        for j in range(num_test) :
            min_dist = distance(test[j], mean[0], train, dist_type)
            min_idx = 0

            for k in range(len(mean)) :
                dist = distance(test[j], mean[k], train, dist_type)
                if dist < min_dist :
                    min_dist = dist
                    min_idx = k

            pred.append(iris_dict[min_idx])
            error.append(min_dist)

            if min_idx == i :
                correct_count += 1
                

    MER = 1.0 - (correct_count / total_num_test)
    
    return MER, correct_count, pred, error


# ### predicting using different distance methods

# In[23]:


MER_vals = []
correct_count_vals = []
error_vals = []

for i in range(dist_nums) :
    MER, correct_count, pred, error = prediction_distance_types(i)
    print("MER for {} distance = {}".format(dist_dict[i], MER))
    print("Number of Correct classifications out of {} = {}\n".format(total_num_test, correct_count))
    MER_vals.append(MER)
    correct_count_vals.append(correct_count)
    error_vals.append(error)


# ### Plotting MER vs distance type classifier

# In[24]:


print("Plotting MER vs Classifier Type")    
plt.bar(list(dist_dict.values()), MER_vals)
plt.xlabel('Classifier Type')
plt.ylabel('MER')

plt.xticks(rotation=90)
plt.yticks(np.arange(0, 0.2, 0.02))
plt.show()


# ### Mean Error and plot

# In[25]:


error_vals = np.array(error_vals)
mean_error = np.mean(error_vals, axis = 1)

print("Plotting Mean Error vs Classifier Type")    
plt.bar(list(dist_dict.values()), mean_error)
plt.xlabel('Classifier Type')
plt.ylabel('Mean Error')

plt.xticks(rotation=90)
plt.show()


# ### Mean Squared Error and plot

# In[26]:


mean_squared_error = np.mean(np.square(error_vals), axis = 1)

print("Plotting Mean Squared Error vs Classifier Type")    
plt.bar(list(dist_dict.values()), mean_squared_error)
plt.xlabel('Classifier Type')
plt.ylabel('Mean Squared Error')

plt.xticks(rotation=90)
plt.show()


# ### Mean absolute Error and plot

# In[27]:


mean_absolute_error = np.mean(np.abs(error_vals), axis = 1)

print("Plotting Mean Absolute Error vs Classifier Type")    
plt.bar(list(dist_dict.values()), mean_absolute_error)
plt.xlabel('Classifier Type')
plt.ylabel('Mean Absolute Error')

plt.xticks(rotation=90)
plt.show()


# In[ ]:




