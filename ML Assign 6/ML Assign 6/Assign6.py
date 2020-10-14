#!/usr/bin/env python
# coding: utf-8

# # Facial Recognition using Artificial Neural Network
# ## submitted by - Harsh Srivastava
# ## Roll - 117CS0755

# ### importing libraries

# In[1]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


# ### displaying some sample images from random folders

# In[2]:


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


# ### creating feature vector for images from training

# In[3]:


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


# ### mean vector

# In[4]:


def create_mean_vec(feature_vector) :

    mean = np.mean(feature_vector, axis = 0)
    #mean_T = np.transpose(mean)
    print("mean vector = ", mean)
    print("shape of mean vector = ", mean.shape)
    print("\n")
    return mean


# ### deviation matrix

# In[5]:


def create_dev_mat(feature_vector, mean) :

    dev_mat = feature_vector - mean
    #dev_mat_T = np.transpose(dev_mat)
    print("deviation matrix = ", dev_mat)
    print("shape of deviation matrix = ", dev_mat.shape)
    print("\n")
    return dev_mat


# ### covariance matrix

# In[6]:


def create_cov_matrix(dev_mat) :

    cov_mat = np.dot(dev_mat, dev_mat.T)
    print("covariance matrix = ", cov_mat)
    print("shape of covariance matrix = ", cov_mat.shape)
    print("\n")
    return cov_mat


# ### eigenvalues and eigenvectors

# In[7]:


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


# In[8]:


def select_k_eigenvectors(eigen_vec, k) :
    dec_feature_vec = eigen_vec[:, :k]
    print("feature vector from pca = ", dec_feature_vec)
    print("shape of selected feature vec = ", dec_feature_vec.shape)
    return dec_feature_vec


# In[9]:


def create_eigen_faces(dec_feature_vec, dev_mat) :
    eig_face = np.dot(np.transpose(dec_feature_vec), dev_mat)
    print("feature vector from pca = ", eig_face)
    print("shape of eigen faces = ", eig_face.shape)
    return eig_face


# In[10]:


def signature_each_face(eig_face, dev_mat) :
    sig_face = np.dot(eig_face, dev_mat.T)
    print("signature of each face = ", sig_face)
    print("dimension of signature = ", sig_face.shape)
    return sig_face


# ### create the test matrix

# In[11]:


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


# ### mean zero

# In[12]:


def create_mean_zero_test(test_mat, mean) :
    
    dev_test_mat = test_mat - mean
    dev_test_mat_T = np.transpose(dev_test_mat)
    print("mean zero test matrix = ", dev_test_mat)
    print("shape of mean zero test matrix = ", dev_test_mat.shape)
    print("\n")
    return dev_test_mat, dev_test_mat_T


# ### actual values of output

# In[13]:


def get_actual_values(num_test) :
    actual_vec = list(range(40))
    actual_vec = np.array([ele for ele in actual_vec for i in range(num_test)])
    print("\nactual expected values\n")
    print(actual_vec)
    print("\nshape of actual output = {}\n".format(actual_vec.shape))
    return actual_vec


# ### important constants

# In[14]:


k = 100 # number of features to be taken from PCA

# number of images in training and test classes
num_train = 6
num_test = 10 - num_train

C = 40 # number of classes/folders


# ### feature vector

# In[15]:


feature_vec = create_feature_vec(num_train)


# ### mean vector

# In[16]:


mean_vec = create_mean_vec(feature_vec)


# ### deviation matrix

# In[17]:


dev_mat = create_dev_mat(feature_vec, mean_vec)


# ### covariance matrix

# In[18]:


cov_mat = create_cov_matrix(dev_mat)


# ### eigen values and vectors

# In[19]:


eigen_val, eigen_vec = create_eig_val_vec(cov_mat)


# ### best direction feature vector

# In[20]:


dec_feature_vec = select_k_eigenvectors(eigen_vec, k)


# ### eigen faces

# In[21]:


eig_face = create_eigen_faces(dec_feature_vec, dev_mat)


# ### signature of faces, i.e., input for ANN

# In[22]:


sig_face = signature_each_face(eig_face, dev_mat)
sig_face = sig_face.T
sig_face.shape


# ### creating the test matrix for testing

# In[23]:


test_mat = create_test_matrix(num_train)


# ### getting deviation test matrix

# In[24]:


dev_test_mat, dev_test_mat_T = create_mean_zero_test(test_mat, mean_vec)


# ###  getting projected test faces

# In[25]:


proj_test_face = np.dot(eig_face, dev_test_mat_T)
print("projected test faces = ", proj_test_face)
print("shape of projected test faces = ", proj_test_face.shape)


# ### input and output matrices for training or building the model

# In[26]:


X = sig_face
y = list(range(C))
y = np.array([ele for ele in y for i in range(num_train)])
print(X)
print(y)
print(X.shape)
print(y.shape)


# ### important constants for ANN model

# In[27]:


num_examples = len(X) # training set size
nn_input_dim = k # input layer dimensionality
nn_output_dim = C # output layer dimensionality

epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength


# ### function to calculate total loss

# In[28]:


def calculate_loss(model):
    # model parameters
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    
    # Add regulatization term to loss (optio
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss


# ### function to predict (1 to 40)

# In[29]:


def predict(model, x):
    # model parameters
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    
    #a1 = np.exp(z1) / 1 + np.exp(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


# ### building model function

# In[30]:


def build_model(nn_hdim, epochs=20000, print_loss=False):
    
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # This is what we return at the end
    model = {}
    
    # Gradient descent. For each batch...
    for i in range(0, epochs):

        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
        
        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
        if print_loss and i % 5000 == 0:
            loss = calculate_loss(model)
            print("Loss after iteration %i: %f" %(i, loss))
    
    return model


# ### accuracy function

# In[31]:


def accuracy(prediction, y):
    count = 0
    for i in range((len(y))) :
        if prediction[i] == y[i] :
            count += 1
    return (count / len(y)) * 100


# ### getting actual output values

# In[32]:


actual_vec = get_actual_values(num_test)


# ### testing for different dimensions of hidden layer

# In[33]:


hidden_layer_dimensions = [1, 5, 20, 50, 100, 200]
acc_val = []
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    print("for {} nodes in hidden layer".format(nn_hdim))
    model = build_model(nn_hdim, print_loss=True)
    prediction = predict(model, proj_test_face.T)
    acc = accuracy(prediction, actual_vec)
    acc_val.append(acc)
    print("accuracy = {} \n".format(acc))


# ### plotting accuracy vs dimensionality of hidden layers

# In[34]:


print("Plotting accuracy vs dimensionality of hidden layers")    
plt.plot(hidden_layer_dimensions, acc_val)
plt.xlabel('Nodes in hidden layer')
plt.ylabel('Accuracy')

plt.yticks(np.arange(0, 100, 10.0))

plt.show()


# In[ ]:




