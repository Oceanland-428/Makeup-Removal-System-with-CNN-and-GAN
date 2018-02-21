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


train_no_dir = '/Users/oceanland/Downloads/E/Stanford/1.2/CS230/project/makeup_with_labels/train/no_makeup/'
train_yes_dir = '/Users/oceanland/Downloads/E/Stanford/1.2/CS230/project/makeup_with_labels/train/yes_makeup/'
val_no_dir = '/Users/oceanland/Downloads/E/Stanford/1.2/CS230/project/makeup_with_labels/val/no_makeup/'
val_yes_dir = '/Users/oceanland/Downloads/E/Stanford/1.2/CS230/project/makeup_with_labels/val/yes_makeup/'

def openFiles(train_no_dir, train_yes_dir, val_no_dir, val_yes_dir):
	train_X, train_Y = create_datasets(train_no_dir, train_yes_dir, True)
	val_X, val_Y = create_datasets(val_no_dir, val_yes_dir)
	return train_X, train_Y, val_X, val_Y

def create_datasets(no_dir, yes_dir, train = False):
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
	count_no = 0
	for i in no_filelist:
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
	count_yes = 0
	for i in yes_filelist:
		img = yes_filelist[i]
		if len(img.shape) == 3 and img.shape[0] == 128 and img.shape[1] == 128 and img.shape[2] == 3:
			X.append(np.array(img))
			Y.append(np.array([1, 0]))
			X_to_yes_file[count_yes] = i
			count_yes += 1
	#if train == True:
	#	X1 = X[:]
	#	Y1 = Y[:]
	#	for i in range(len(X1)):
	#		X1[i] -= 3
	#	X += X1
	#	Y += Y1
	X = np.array(X)
	Y = np.array(Y)
	return X, Y

def create_placeholders(n_H0, n_W0, n_C0, n_y):
	X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
	Y = tf.placeholder(tf.float32, [None, n_y])
	return X, Y

def initialize_parameters():
	W1 = tf.get_variable("W1", [3, 3, 3, 8], initializer = tf.contrib.layers.xavier_initializer())
	W2 = tf.get_variable("W2", [3, 3, 8, 16], initializer = tf.contrib.layers.xavier_initializer())
	W3 = tf.get_variable("W3", [3, 3, 16, 32], initializer = tf.contrib.layers.xavier_initializer())
	W4 = tf.get_variable("W4", [3, 3, 32, 64], initializer = tf.contrib.layers.xavier_initializer())

	parameters = {"W1": W1, "W2": W2, "W3": W3, "W4": W4}

	return parameters

def forward_propagation(X, parameters):
	W1 = parameters['W1']
	W2 = parameters['W2']
	W3 = parameters['W3']
	W4 = parameters['W4']

	Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME')
	A1 = tf.nn.relu(Z1)
	Z2 = tf.nn.conv2d(A1,W2, strides = [1,1,1,1], padding = 'SAME')
	A2 = tf.nn.relu(Z2)
	P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')

	Z3 = tf.nn.conv2d(P2,W3, strides = [1,1,1,1], padding = 'SAME')
	A3 = tf.nn.relu(Z3)
	Z4 = tf.nn.conv2d(A3,W4, strides = [1,1,1,1], padding = 'SAME')
	A4 = tf.nn.relu(Z4)
	P4 = tf.nn.max_pool(A4, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')

	#Z3 = tf.nn.conv2d(P2,W3, strides = [1,1,1,1], padding = 'SAME')
	#A3 = tf.nn.relu(Z3)
	#P3 = tf.nn.max_pool(A3, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')

	#Z4 = tf.nn.conv2d(P3,W4, strides = [1,1,1,1], padding = 'SAME')
	#A4 = tf.nn.relu(Z4)
	#P4 = tf.nn.max_pool(A4, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')

	P4 = tf.contrib.layers.flatten(P4)
	Z5 = tf.contrib.layers.fully_connected(P4, 2, activation_fn=None)
	return Z5

def compute_cost(Z, Y):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels = Y))
	return cost

def model(train_X, train_Y, val_X, val_Y, learning_rate = 0.015, num_epochs = 100, minibatch_size = 64, print_cost = True):

	ops.reset_default_graph()                                          
	(m, n_H0, n_W0, n_C0) = train_X.shape             
	n_y = train_Y.shape[1]                            
	costs = []

	X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

	parameters = initialize_parameters()

	Z = forward_propagation(X, parameters)

	cost = compute_cost(Z, Y)

	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:

		sess.run(init)

		for epoch in range(num_epochs):

			_ , epoch_cost = sess.run([optimizer, cost], feed_dict={X: train_X, Y: train_Y})
			#print Z

			if print_cost == True and epoch % 5 == 0:
				print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
			if print_cost == True and epoch % 1 == 0:
				costs.append(epoch_cost)

		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per tens)')
		plt.title("Learning rate =" + str(learning_rate))
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





def main():
	train_X, train_Y, val_X, val_Y = openFiles(train_no_dir, train_yes_dir, val_no_dir, val_yes_dir)
	print train_X.shape	# (835, 128, 128, 3)
	print train_Y.shape	# (835, 2)
	print val_X.shape	# (200, 128, 128, 3)
	print val_Y.shape	# (200, 2)
	_, _, parameters = model(train_X, train_Y, val_X, val_Y)






if __name__ == '__main__':
	main()
