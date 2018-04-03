import sys
import tensorflow as tf
import collections
from layers import *

batch_size = 16

class Model(object):

    def __init__(self, name):
        self.name = name

    def get_trainable_variables(self):
        t_vars = tf.trainable_variables()
        t_vars_model = {v.name: v for v in t_vars if self.name in v.name}
        return t_vars_model


class Generator(Model):
    def __init__(self, name="generator", is_training = True):
        super(Generator, self).__init__(name)
        self.is_training = is_training
        self.start_dim = int(self.output_h / 16)
        self.nb_upconv = 4
        self.filters = 512

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = layers.linear(x, self.filters * self.start_dim * self.start_dim)
            target_shape = (batch_size, self.start_dim, self.start_dim, self.filters)
            x = layers.reshape(x, target_shape)
            x = tf.contrib.layers.batch_norm(x, fused=True)
            x = tf.nn.relu(x)
            for idx, (f, k, s, p) in enumerate(zip(self.list_filters, self.list_kernel_size, self.list_strides, self.list_padding)):
                name = "upsample2D_%s" % idx
                if idx == len(self.list_filters) - 1:
                    bn = False
                    activation_fn = None
                else:
                    bn = True
                    activation_fn = tf.nn.relu
                x = layers.upsample2d_block(name, x, f, k, s, p, data_format=self.data_format, bn=bn, activation_fn=activation_fn)

            x = tf.nn.tanh(x, name="X_G")
            return x


class Discriminator(Model):
    def __init__(self, name="discriminator", is_training = True):
        super(Discriminator, self).__init__(name)
        self.is_training = is_training

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            for idx, (f, k, s, p) in enumerate(zip(self.list_filters, self.list_kernel_size, self.list_strides, self.list_padding)):
                if idx == 0:
                    bn = False
                else:
                    bn = True
                name = "conv2D_%s" % idx
                x = layers.conv2d_block(name, x, f, k, s, p=p, stddev=0.02,
                                        data_format=self.data_format, bias=True, bn=bn, activation_fn=layers.lrelu)

            target_shape = (self.batch_size, -1)
            x = layers.reshape(x, target_shape)

            # # Add MBD
            # x_mbd = layers.mini_batch_disc(x, num_kernels=100, dim_per_kernel=5)
            # # Concat
            # x = tf.concat([x, x_mbd], axis=1)

            x = layers.linear(x, 1, bias=False)

            return x











