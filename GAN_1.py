import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from layers import *
from extract_encoding import *
%matplotlib inline


def create_datasets(no_dir, yes_dir):
	no_files = glob.glob(no_dir + '*.jpg')
	no_filelist = []
	X = []
	Y = []
	for file in no_files:
		img = Image.open(file)
		arr = np.array(img)
		no_filelist.append(arr)
		img_flip = np.fliplr(img)
		arr_flip = np.array(img_flip)
		no_filelist.append(arr_flip)
	yes_files = glob.glob(yes_dir + '*.jpg')
	yes_filelist = []
	for file in yes_files:
		img = Image.open(file)
		arr = np.array(img)
		yes_filelist.append(arr)
		img_flip = np.fliplr(img)
		arr_flip = np.array(img_flip)
		yes_filelist.append(arr_flip)
	for i in range(len(no_filelist)):
		img_no = no_filelist[i]
		img_yes = yes_filelist[i]
		if len(img_no.shape) == 3 and len(img_yes) == 3 and img_no.shape[2] == 3 and img_yes.shape[2] == 3:
			X.append(np.array(img_yes))
			Y.append(np.array(img_no))
	X = np.array(X)
	Y = np.array(Y)
	return X, Y

def avg_pool_2x2(x):
	return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def discriminator(x, reuse=False):
	with tf.variable_scope('discriminator'):
		if (reuse):
			tf.get_variable_scope().reuse_variables()
		#First Conv and Pool Layers
		x = lrelu_layer(conv2d_layer(x, 64, 4, 4, 2, 2, name='d_conv1'))
		x = lrelu_layer(bn_layer(conv2d_layer(x, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
		x = lrelu_layer(bn_layer(conv2d_layer(x, 256, 4, 4, 2, 2, name='d_conv3'), is_training=is_training, scope='d_bn3'))
		x = lrelu_layer(bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='d_conv4'), is_training=is_training, scope='d_bn4'))
		x = lrelu_layer(bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='d_conv5'), is_training=is_training, scope='d_bn5'))
		x = lrelu_layer(bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='d_conv6'), is_training=is_training, scope='d_bn6'))
		x = conv2d_layer(x, 1, 4, 4, 1, 1, name='d_conv7')
		x = tf.reshape(x, [self.batch_size, -1]) # Can use tf.reduce_mean(x, axis=[1, 2, 3])
		x = linear_layer(x, 1, scope='d_fc8')

		return x

def generator(x, is_training=True, reuse=False):
	with tf.variable_scope("generator"):
		conv_layer = []
		x = lrelu_layer(conv2d_layer(x, 64, 4, 4, 2, 2, name='g_conv1'))
		conv_layer.append(x)
		x = lrelu_layer(bn_layer(conv2d_layer(x, 128, 4, 4, 2, 2, name='g_conv2'), is_training=is_training, scope='g_bn2'))
		conv_layer.append(x)
		x = lrelu_layer(bn_layer(conv2d_layer(x, 256, 4, 4, 2, 2, name='g_conv3'), is_training=is_training, scope='g_bn3'))
		conv_layer.append(x)
		x = lrelu_layer(bn_layer(conv2d_layer(x, 256, 4, 4, 2, 2, name='g_conv4'), is_training=is_training, scope='g_bn4'))
		conv_layer.append(x)
		x = lrelu_layer(bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='g_conv5'), is_training=is_training, scope='g_bn5'))
		conv_layer.append(x)
		x = lrelu_layer(bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='g_conv6'), is_training=is_training, scope='g_bn6'))

		x = lrelu_layer(bn_layer(deconv2d_layer(x, 512, 4, 4, 2, 2, name='g_dconv1'), is_training=is_training, scope='gd_bn1'))
		x = tf.concat([x, conv_layer.pop()], axis=3)
		x = lrelu_layer(bn_layer(deconv2d_layer(x, 256, 4, 4, 2, 2, name='g_dconv2'), is_training=is_training, scope='gd_bn2'))
		x = tf.concat([x, conv_layer.pop()], axis=3)
		x = lrelu_layer(bn_layer(deconv2d_layer(x, 256, 4, 4, 2, 2, name='g_dconv3'), is_training=is_training, scope='gd_bn3'))
		x = tf.concat([x, conv_layer.pop()], axis=3)
		x = lrelu_layer(bn_layer(deconv2d_layer(x, 128, 4, 4, 2, 2, name='g_dconv4'), is_training=is_training, scope='gd_bn4'))
		x = tf.concat([x, conv_layer.pop()], axis=3)
		x = lrelu_layer(bn_layer(deconv2d_layer(x, 64, 4, 4, 2, 2, name='g_dconv5'), is_training=is_training, scope='gd_bn5'))
		x = tf.concat([x, conv_layer.pop()], axis=3)
		x = lrelu_layer(bn_layer(deconv2d_layer(x, 3, 4, 4, 2, 2, name='g_dconv6'), is_training=is_training, scope='gd_bn6'))
		x = tf.tanh(x)

		return x

def create_placeholders(n_H0, n_W0, n_C0):
	X_with = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
	X_without = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
	return X_with, X_without

def nextBatch(X, Y, num, batch_size):
	if (num + 1) * batch_size >= Y.shape[0]:
		return X[num * batch_size:,:,:,:], Y[num * batch_size:,:,:,:]
	else:
		return X[num * batch_size:(num + 1) * batch_size,:,:,:], Y[num * batch_size:(num + 1) * batch_size,:,:,:]



no_dir = 'gan_makeup_data_96/without/'
yes_dir = 'gan_makeup_data_96/with/'

X, Y = create_datasets(no_dir, yes_dir)
print X.shape # (3961, 128, 128, 3)
print Y.shape # (3961, 2)
(m, n_H0, n_W0, n_C0) = train_X.shape 
n_y = train_Y.shape[1]
X_with, X_without = create_placeholders(n_H0, n_W0, n_C0)


batch_size = 16
num_batch = m / batch_size + 1

tf.reset_default_graph()

sess = tf.Session()

D_real = discriminator(X_with)
Gz = generator(X_without)
D_fake = discriminator(Gz, reuse=True)

g_loss_bin = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake, labels = tf.ones_like(D_fake)))
g_loss_l1 = computeL1(Gz, X_without) #	TODO: extract encodings
g_loss = g_loss_bin + g_loss_l1

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real, labels = tf.ones_like(D_real)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake, labels = tf.zeros_like(D_fake)))
d_loss = d_loss_real + d_loss_fake


tvars = tf.trainable_variables()
d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")

print(tf.get_variable_scope().reuse)
optimizerD = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_vars)
optimizerG = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_vars)

sess.run(tf.global_variables_initializer())
iterations = 3000
for i in range(iterations):
	for num in range(num_batch):
		with_makeup_batch, without_makeup_batch = nextBatch(X, Y, num, batch_size) 
		_,dLoss = sess.run([optimizerD, d_loss],feed_dict={X_with: with_makeup_batch, X_without: without_makeup_batch})
		_,gLoss = sess.run([optimizerG, g_loss],feed_dict={X_with: with_makeup_batch, X_without: without_makeup_batch})
	if i % 100 == 0:
		print dLoss, gLoss
		sample = sess.run(Gz, feed_dict={X_without: X[0]})
		print X[0]
		print Y[0]
		print sample
	#train_X, train_Y, val_X, val_Y = shuffleData(train_X, train_Y, val_X, val_Y)

'''
sample_image = generator(z_placeholder, 1, z_dimensions)
z_batch = np.random.normal(-1, 1, size=[1, z_dimensions])
temp = (sess.run(sample_image, feed_dict={z_placeholder: z_batch}))
my_i = temp.squeeze()
plt.imshow(my_i, cmap='gray_r')
'''



