import sys
import tensorflow as tf
import collections
from layers_old import *
sys.path.append("../utils")
import layers as ly


class Model(object):

    def __init__(self, name):
        self.name = name

    def get_trainable_variables(self):
        t_vars = tf.trainable_variables()
        t_vars_model = {v.name: v for v in t_vars if self.name in v.name}
        return t_vars_model


class Generator(Model):
    def __init__(self, list_filters, list_kernel_size, list_strides, list_padding, output_shape,
                 name="generator", batch_size=32, filters=512, dset="celebA", data_format="NCHW"):

        super(Generator, self).__init__(name)

        self.data_format = data_format

        if self.data_format == "NCHW":
            self.output_h = output_shape[1]
            self.output_w = output_shape[2]
        else:
            self.output_h = output_shape[0]
            self.output_w = output_shape[1]

        if dset == "mnist":
            self.start_dim = int(self.output_h / 4)
            self.nb_upconv = 2
        else:
            self.start_dim = int(self.output_h / 16)
            self.nb_upconv = 4

        self.output_shape = output_shape
        self.dset = dset
        self.name = name
        self.batch_size = batch_size
        self.filters = filters
        self.list_filters = list_filters
        self.list_kernel_size = list_kernel_size
        self.list_padding = list_padding
        self.list_strides = list_strides
        self.is_training = True

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:

            if reuse:
                scope.reuse_variables()
            '''
            # Store all layers in a dict
            d = collections.OrderedDict()

            # Initial dense multiplication
            x = ly.linear(x, self.filters * self.start_dim * self.start_dim)

            # Reshape to image format
            if self.data_format == "NCHW":
                target_shape = (self.batch_size, self.filters, self.start_dim, self.start_dim)
            else:
                target_shape = (self.batch_size, self.start_dim, self.start_dim, self.filters)

            x = ly.reshape(x, target_shape)
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
                x = ly.upsample2d_block(name, x, f, k, s, p, data_format=self.data_format, bn=bn, activation_fn=activation_fn)

            x = tf.nn.tanh(x, name="X_G")
            
            '''
            conv_layer = []
            #print (x)
            x = lrelu_layer(conv2d_layer(x, 64, 4, 4, 2, 2, name='g_conv1'))
            conv_layer.append(x)
            #print ('x',x)
            x = conv2d_layer(x, 128, 4, 4, 2, 2, name='g_conv2')
            #print ('x',x)
            x = bn_layer(x, is_training=self.is_training, scope='g_bn_test')
            x = lrelu_layer(x)
            conv_layer.append(x)
            x = lrelu_layer(bn_layer(conv2d_layer(x, 256, 4, 4, 2, 2, name='g_conv3'), is_training=self.is_training, scope='g_bn3'))
            conv_layer.append(x)
            x = lrelu_layer(bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='g_conv4'), is_training=self.is_training, scope='g_bn4'))
            conv_layer.append(x)
            x = lrelu_layer(bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='g_conv5'), is_training=self.is_training, scope='g_bn5'))
            x = lrelu_layer(bn_layer(deconv2d_layer(x, 512, 4, 4, 2, 2, name='g_dconv4'), is_training=self.is_training, scope='gd_bn4'))
            x = tf.concat([x, conv_layer.pop()], axis=3)
            x = lrelu_layer(bn_layer(deconv2d_layer(x, 256, 4, 4, 2, 2, name='g_dconv5'), is_training=self.is_training, scope='gd_bn5'))
            x = tf.concat([x, conv_layer.pop()], axis=3)
            x = lrelu_layer(bn_layer(deconv2d_layer(x, 128, 4, 4, 2, 2, name='g_dconv6'), is_training=self.is_training, scope='gd_bn6'))
            x = tf.concat([x, conv_layer.pop()], axis=3)
            x = lrelu_layer(bn_layer(deconv2d_layer(x, 64, 4, 4, 2, 2, name='g_dconv7'), is_training=self.is_training, scope='gd_bn7'))
            x = tf.concat([x, conv_layer.pop()], axis=3)
            x = lrelu_layer(bn_layer(deconv2d_layer(x, 3, 4, 4, 2, 2, name='g_dconv8'), is_training=self.is_training, scope='gd_bn8'))
            x = tf.tanh(x)
            
            return x


class Discriminator(Model):
    def __init__(self, list_filters, list_kernel_size, list_strides, list_padding, batch_size,
                 name="discriminator", data_format="NCHW"):
        # Determine data format from output shape

        super(Discriminator, self).__init__(name)

        self.data_format = data_format
        self.name = name
        self.list_filters = list_filters
        self.list_strides = list_strides
        self.list_kernel_size = list_kernel_size
        self.batch_size = batch_size
        self.list_padding = list_padding

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
                x = ly.conv2d_block(name, x, f, k, s, p=p, stddev=0.02,
                                        data_format=self.data_format, bias=True, bn=bn, activation_fn=ly.lrelu)

            target_shape = (self.batch_size, -1)
            x = ly.reshape(x, target_shape)

            # # Add MBD
            # x_mbd = layers.mini_batch_disc(x, num_kernels=100, dim_per_kernel=5)
            # # Concat
            # x = tf.concat([x, x_mbd], axis=1)

            x = ly.linear(x, 1, bias=False)

            return x
