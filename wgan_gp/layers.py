import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers


def lrelu_layer(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak*x)

def bn_layer(x, is_training, scope):
	return layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training, scope=scope)

def conv2d_layer(x, num_filters, filter_height, filter_width, stride_height, stride_width, stddev=0.2, name="conv2d"):
	with tf.variable_scope(name):
		w = tf.get_variable('weight', [filter_height, filter_width, x.get_shape()[-1], num_filters], initializer=tf.truncated_normal_initializer(stddev=stddev)) #weights
		s = [1, stride_height, stride_width, 1] # stride

		conv = tf.nn.conv2d(x, w, s, padding='SAME')
		biases = tf.get_variable('bias', [num_filters], initializer=tf.constant_initializer(0.0))
		conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

		return conv

def deconv2d_layer(x, out_channel, filter_height, filter_width, stride_height, stride_width, stddev=0.2, name="deconv2d"):
	with tf.variable_scope(name):
		in_channel = x.get_shape()[-1]
		out_shape = [int(x.get_shape()[0]), int(x.get_shape()[1]*stride_height), int(x.get_shape()[2]*stride_width), out_channel]
		#out_shape = tf.convert_to_tensor(out_shape)
		w = tf.get_variable("weight", [filter_height, filter_width, out_channel, x.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
		s = [1, stride_height, stride_width, 1]
		deconv = tf.nn.conv2d_transpose(x, w, out_shape, s, padding='SAME')
		biases = tf.get_variable('bias', out_channel, initializer=tf.constant_initializer(0.0))
		deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

		return deconv

def linear_layer(x, output_size, scope=None, stddev=0.2, bias_start=0.0, with_w=False):
	shape = x.get_shape().as_list()

	with tf.variable_scope(scope or "Linear"):
		matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
		bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
		if with_w: # return values along with parameters of fc_layer
			return tf.matmul(x, matrix) + bias, matrix, bias
		else:
			return tf.matmul(x, matrix) + bias


			