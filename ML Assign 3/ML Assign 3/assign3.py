#!/usr/bin/env python
# coding: utf-8

# In[18]:


# importing libraries

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


# In[19]:


# displaying some sample images from random folders

base_path = 'facial_data/Datasets/att_faces_combined/'
    
img=cv2.imread(base_path + 's1/1.pgm')
imgplot = plt.imshow(img)
plt.show()
    
img=cv2.imread(base_path + 's2/3.pgm')
imgplot = plt.imshow(img)
plt.show()
    
img=cv2.imread(base_path + 's3/5.pgm')
imgplot = plt.imshow(img)
plt.show()
    
img=cv2.imread(base_path + 's4/9.pgm')
imgplot = plt.imshow(img)
plt.show()

img=cv2.imread(base_path + 's5/6.pgm')
imgplot = plt.imshow(img)
plt.show()

leny, lenx, z = img.shape


# In[20]:


# creating feature vector for images from training
      
def create_feature_vec(num_train) :  

    feature_vector = []
    for i in range(1, 41) : # for each folder
        folder_path = base_path + 's' + str(i) + '/'
        for j in range(1, num_train + 1) : # for each image inside the folder si
            img_path = folder_path + str(j) + '.pgm'
            im = Image.open(img_path)
            pix = im.load()

            img_row = []
            for x in range(lenx) :
                for y in range(leny) :
                    img_row.append(pix[x, y])

            feature_vector.append(img_row)

    feature_vector = np.array(feature_vector)
    feature_vector_T = np.transpose(feature_vector)
    print("feature vector = ", feature_vector)
    print("shape of feature vector = ", feature_vector.shape)
    print("\n")

    return feature_vector


# In[21]:


# mean vector

def create_mean_vec(feature_vector) :

    mean = np.mean(feature_vector, axis = 0)
    mean_T = np.transpose(mean)
    print("mean vector = ", mean)
    print("shape of mean vector = ", mean.shape)
    print("\n")
    return mean, mean_T


# In[22]:


# deviation matrix

def create_dev_mat(feature_vector, mean) :

    dev_mat = feature_vector - mean
    dev_mat_T = np.transpose(dev_mat)
    print("deviation matrix = ", dev_mat)
    print("shape of deviation matrix = ", dev_mat.shape)
    print("\n")
    return dev_mat, dev_mat_T


# In[23]:


# covariance matrix

def create_cov_matrix(dev_mat, dev_mat_T) :

    cov_mat = np.dot(dev_mat, dev_mat_T)
    print("covariance matrix = ", cov_mat)
    print("shape of covariance matrix = ", cov_mat.shape)
    print("\n")
    return cov_mat


# In[24]:


# eigenvalues and eigenvectors

def create_eig_val_vec(cov_mat) :

    eigen_val, eigen_vec = np.linalg.eig(cov_mat)

    eigen_val_s = np.sort(eigen_val)
    eigen_vec_s = eigen_vec[:, eigen_val.argsort()]
    eigen_vec_s = np.fliplr(eigen_vec_s)

    eigen_vec = eigen_vec_s
    eigen_val = eigen_val_s
    
    print("shape of eigen values vector -->",eigen_val.shape)
    print("shape of eigen vector matrix -->",eigen_vec.shape)
    print("\n")

    return eigen_val, eigen_vec


# In[25]:


def select_k_eigenvectors(eigen_vec, k) :
    return eigen_vec[:, :k]


# In[26]:


def create_eigen_faces(dec_feature_vec, dev_mat) :
    eig_face = np.dot(np.transpose(dec_feature_vec), dev_mat)
    return eig_face


# In[27]:


def signature_each_face(eig_face, dev_mat_T) :
    sig_face = np.dot(eig_face, dev_mat_T)
    return sig_face


# In[28]:


def prediction_vec(proj_test_face, sig_face, k, num_test, num_train) :    
    min_idx_each_img = []

    for i in range(num_test * 40) : # for each test image
        test_img_proj = proj_test_face[:, i]
        test_img_proj = np.reshape(test_img_proj, (k, 1))

        dist_vec = []

        for j in range(num_train * 40) : # for each signature of training images
            sig_each_face = sig_face[:, j]
            sig_each_face = np.reshape(sig_each_face, (k, 1))
            dist_vec.append(euclidean_dist(test_img_proj, sig_each_face))

        dist_vec = np.array(dist_vec)
        #print(dist_vec)
        min_dist = dist_vec[0]
        min_idx = 0

        for j in range(len(dist_vec)) :
            if dist_vec[j] < min_dist :
                min_dist = dist_vec[j]
                min_idx = j
        min_idx_each_img.append(min_idx // num_train + 1) 
    return min_idx_each_img


# In[29]:


def euclidean_dist(a, b) :
    dist = a - b
    sq_dist = np.dot(np.transpose(dist), dist)
    sq_dist = np.sqrt(sq_dist)
    return sq_dist


# In[30]:


# creating testing vector
def create_test_vec(num_test) :

    test_vec = list(range(1, 41))
    test_vec =  [ele for ele in test_vec for i in range(num_test)] 

    print("expected values = ",test_vec)
    print("\n")

    return test_vec


# In[31]:


# displaying some sample images from random folders
    
img=cv2.imread(base_path + 's1/9.pgm')
imgplot = plt.imshow(img)
plt.show()
    
img=cv2.imread(base_path + 's1/10.pgm')
imgplot = plt.imshow(img)
plt.show()
    
img=cv2.imread(base_path + 's3/10.pgm')
imgplot = plt.imshow(img)
plt.show()
    
img=cv2.imread(base_path + 's4/10.pgm')
imgplot = plt.imshow(img)
plt.show()

img=cv2.imread(base_path + 's5/10.pgm')
imgplot = plt.imshow(img)
plt.show()


# In[32]:


# create the test matrix
    
def create_test_matrix(num_train) :
    
    test_mat = []
    for i in range(1, 41) : # for each folder
        folder_path = base_path + 's' + str(i) + '/'
        for j in range(num_train + 1, 11) :

            img_path = folder_path + str(j) + '.pgm'
            im = Image.open(img_path)
            pix = im.load()

            img_row = []
            for x in range(lenx) :
                for y in range(leny) :
                    img_row.append(pix[x, y])

            test_mat.append(img_row)

    test_mat = np.array(test_mat)
    print("test matrix = ", test_mat)
    print("shape of test matrix = ", test_mat.shape)
    print("\n")
    return test_mat        


# In[33]:


# mean zero

def create_mean_zero_test(test_mat, mean) :
    
    dev_test_mat = test_mat - mean
    dev_test_mat_T = np.transpose(dev_test_mat)
    print("mean zero test matrix = ", dev_test_mat)
    print("shape of mean zero test matrix = ", dev_test_mat.shape)
    print("\n")
    return dev_test_mat, dev_test_mat_T


# In[34]:


# final testing

def testing_for_k(k, eigen_vec, dev_mat, dev_mat_T, dev_test_mat_T, num_test, num_train) :
    
    dec_feature_vec = select_k_eigenvectors(eigen_vec, k)
    
    eig_face = create_eigen_faces(dec_feature_vec, dev_mat)
    
    sig_face = signature_each_face(eig_face, dev_mat_T)
    
    proj_test_face = np.dot(eig_face, dev_test_mat_T)
    
    min_idx_each_img = prediction_vec(proj_test_face, sig_face, k, num_test, num_train)
    
    return min_idx_each_img


# In[35]:


# calculating accuracy

def accuracy(min_idx_each_img, test_vec, num_test) :
    count = 0
    for i in range(num_test * 40) :
        if min_idx_each_img[i] == test_vec[i] :
            count = count + 1
    return (count / (num_test * 40)) * 100


# In[36]:


# output for various train-test split cases

def diff_cases(num_train, num_test) :
    
    # getting feature vector
    feature_vector = create_feature_vec(num_train)
    
    # getting mean
    mean, mean_T = create_mean_vec(feature_vector)
    
    # getting deaviation matrix
    dev_mat, dev_mat_T = create_dev_mat(feature_vector, mean)
    
    # getting covariance matrix
    cov_mat = create_cov_matrix(dev_mat, dev_mat_T)
    
    # getting eigen values and eigen vectors
    eigen_val, eigen_vec = create_eig_val_vec(cov_mat)
    
    # getting test matrix
    test_mat = create_test_matrix(num_train)
    
    # getting deviation test matrix
    dev_test_mat, dev_test_mat_T = create_mean_zero_test(test_mat, mean)
    
    # getting actual test values
    test_vec = create_test_vec(num_test)
    
    # accuracy for k values
    
    k_val = []
    acc_val = []
    for k in range(51) :
        min_idx_each_img = testing_for_k(k, eigen_vec, dev_mat, dev_mat_T, dev_test_mat_T, num_test, num_train)
        
        k_val.append(k)
        acc_val.append(accuracy(min_idx_each_img, test_vec, num_test)) 
    
    print("Plotting accuracy vs k")    
    plt.plot(k_val, acc_val)
    plt.xlabel('Increasing value of k')
    plt.ylabel('Accuracy')

    plt.xticks(np.arange(min(k_val), max(k_val) + 1, 5.0))
    plt.yticks(np.arange(0, 100, 10.0))

    plt.show()
    
    print("maximum accuracy for {}-{} split = {}".format(num_train * 10, num_test * 10, max(acc_val)))


# In[37]:


diff_cases(8, 2)


# In[38]:


diff_cases(9, 1)


# In[39]:


diff_cases(6, 4)


# In[ ]:





# In[ ]:




