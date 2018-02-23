
# coding: utf-8

# In[47]:


import tensorflow as tf
from PIL import Image
import glob
import numpy as np
import math
import h5py
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from tensorflow.python.framework import ops
np.random.seed(1)


# In[48]:


no_dir = 'makeup_with_labels/no_makeup/'
yes_dir = 'makeup_with_labels/yes_makeup/'


# In[49]:


def create_datasets(no_dir, yes_dir):
	no_files = glob.glob(no_dir + '*.jpg')
	no_filelist = []
	X = []
	Y = []
	X_to_no_file = {}
	X_to_yes_file = {}
	for file in no_files:
		img = Image.open(file)
		arr = np.array(img)
		no_filelist.append(arr)
		img_flip = np.fliplr(img)
		arr_flip = np.array(img_flip)
		no_filelist.append(arr_flip)
	count_no = 0
	for i in range(len(no_filelist)):
		img = no_filelist[i]
		if len(img.shape) == 3 and img.shape[0] == 128 and img.shape[1] == 128 and img.shape[2] == 3:
			X.append(np.array(img))
			Y.append(np.array([0, 1]))
			X_to_no_file[count_no] = i
			count_no += 1
	yes_files = glob.glob(yes_dir + '*.jpg')
	yes_filelist = []
	for file in yes_files:
		img = Image.open(file)
		arr = np.array(img)
		yes_filelist.append(arr)
		img_flip = np.fliplr(img)
		arr_flip = np.array(img_flip)
		yes_filelist.append(arr_flip)
	count_yes = 0
	for i in range(len(yes_filelist)):
		img = yes_filelist[i]
		if len(img.shape) == 3 and img.shape[0] == 128 and img.shape[1] == 128 and img.shape[2] == 3:
			X.append(np.array(img))
			Y.append(np.array([1, 0]))
			X_to_yes_file[count_yes] = i
			count_yes += 1
	X = np.array(X)
	Y = np.array(Y)
	s = np.arange(X.shape[0])
	shuffled_X = np.copy(X)[s]
	shuffled_Y = np.copy(Y)[s]
	num_data = shuffled_Y.shape[0]
	train_X = np.copy(shuffled_X[:int(num_data*0.8)])
	val_X = np.copy(shuffled_X[int(num_data*0.8):])
	train_Y = np.copy(shuffled_Y[:int(num_data*0.8)])
	val_Y = np.copy(shuffled_Y[int(num_data*0.8):])
	return train_X, train_Y, val_X, val_Y


# In[50]:


def create_placeholders(n_H0, n_W0, n_C0, n_y):
	X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
	Y = tf.placeholder(tf.float32, [None, n_y])
	return X, Y


# In[51]:


def initialize_parameters():
    tf.set_random_seed(1)
    W1 = tf.get_variable("W1", [3, 3, 3, 8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", [3, 3, 8, 16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W3 = tf.get_variable("W3", [3, 3, 16, 32], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W4 = tf.get_variable("W4", [3, 3, 32, 64], initializer = tf.contrib.layers.xavier_initializer(seed = 0))

    parameters = {"W1": W1, "W2": W2, "W3": W3, "W4": W4}

    return parameters


# In[52]:


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']

    Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME')
    A1 = tf.nn.relu(Z1)
    #Z2 = tf.nn.conv2d(A1,W2, strides = [1,1,1,1], padding = 'SAME')
    #A2 = tf.nn.relu(Z2)
    P1 = tf.nn.max_pool(A1, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')

    Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    #Z4 = tf.nn.conv2d(A3,W4, strides = [1,1,1,1], padding = 'SAME')
    #A4 = tf.nn.relu(Z4)
    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')

    #Z3 = tf.nn.conv2d(P2,W3, strides = [1,1,1,1], padding = 'SAME')
    #A3 = tf.nn.relu(Z3)
    #P3 = tf.nn.max_pool(A3, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')

    #Z4 = tf.nn.conv2d(P3,W4, strides = [1,1,1,1], padding = 'SAME')
    #A4 = tf.nn.relu(Z4)
    #P4 = tf.nn.max_pool(A4, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')

    P3 = tf.contrib.layers.flatten(P2)
    A4 = tf.contrib.layers.fully_connected(P3, 512)
    #dropout = tf.layers.dropout(inputs = A4, rate=0.4)
    Z5 = tf.contrib.layers.fully_connected(A4, 2, activation_fn=None)
    return Z5


# In[53]:


def compute_cost(Z, Y, parameters, lambd, regu = False):
    if regu:
        print "L2 regu"
        # add regularization
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels = Y))
        regularizer = tf.nn.l2_loss(parameters["W1"]) + tf.nn.l2_loss(parameters["W2"]) + tf.nn.l2_loss(parameters["W3"]) + tf.nn.l2_loss(parameters["W4"])
        cost = tf.reduce_mean(cost + lambd * regularizer)
        return cost
    else:
        print "no L2 regu"
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels = Y))
        return cost


# In[54]:


def model(train_X, train_Y, val_X, val_Y, learning_rate, num_epochs, lambd, regu, print_cost = True):

    ops.reset_default_graph()  
    tf.set_random_seed(1)
    (m, n_H0, n_W0, n_C0) = train_X.shape             
    n_y = train_Y.shape[1]                            
    costs = []
    val_costs = []

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    parameters = initialize_parameters()

    Z = forward_propagation(X, parameters)

    cost = compute_cost(Z, Y, parameters, lambd, regu)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(num_epochs):

            _ , epoch_cost = sess.run([optimizer, cost], feed_dict={X: train_X, Y: train_Y})
            #print Z

            if print_cost == True and epoch % 5 == 0:
                print "Cost after epoch %i: %f" % (epoch, epoch_cost)
            if print_cost == True and epoch % 1 == 0:
                costs.append(epoch_cost)
            if print_cost == True and epoch % 50 == 0:
                val_epoch_cost = sess.run(cost, feed_dict={X: val_X, Y: val_Y})
                val_costs.append(val_epoch_cost)
                print "Val cost after %i: %f" % (epoch, val_epoch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("LR: " + str(learning_rate) + "LD: " + str(lambd))
        plt.show()
        plt.plot(np.squeeze(val_costs))
        plt.ylabel('val_costs')
        plt.xlabel('iterations (per 50s)')
        plt.title("LR: " + str(learning_rate) + "LD: " + str(lambd))
        plt.show()

        predict_op = tf.argmax(Z, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: train_X, Y: train_Y})
        test_accuracy = accuracy.eval({X: val_X, Y: val_Y})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, parameters


# In[55]:


def main():
    train_X, train_Y, val_X, val_Y = create_datasets(no_dir, yes_dir)
    print train_X.shape # (1670, 128, 128, 3)
    print train_Y.shape # (1670, 2)
    print val_X.shape   # (200, 128, 128, 3)
    print val_Y.shape   # (200, 2)
    n_epoch = 1000
    lambd_list = [1e5, 1e4, 1e3, 1e2, 1e1, 1]
    regu = True
    learning_rate_dict = {}
    for i in range(1):
        r = np.random.rand()
        r = - r - 2
        learning_rate = 10 ** r
        learning_rate = 0.003
        #print "learning_rate", learning_rate
        for lambd in lambd_list:
            train_accuracy, test_accuracy, parameters = model(train_X, train_Y, val_X, val_Y, learning_rate, n_epoch, 1, regu)
            learning_rate_dict[learning_rate] = (train_accuracy, test_accuracy, parameters)
    for lr in learning_rate_dict:
        print lr, learning_rate_dict[lr][0], learning_rate_dict[lr][1]


# In[ ]:


if __name__ == '__main__':
	main()

