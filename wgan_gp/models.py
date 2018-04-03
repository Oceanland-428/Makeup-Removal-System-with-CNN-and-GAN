import sys
import tensorflow as tf
import collections
import layers

class Model(object):

    def __init__(self, name):
        self.name = name

    def get_trainable_variables(self):
        t_vars = tf.trainable_variables()
        t_vars_model = {v.name: v for v in t_vars if self.name in v.name}
        return t_vars_model


class Generator(Model):
    def __init__(self, name="generator"):
        super(Generator, self).__init__(name)

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            conv_layer = []
            print (x)
            x = lrelu_layer(conv2d_layer(x, 64, 4, 4, 2, 2, name='g_conv1'))
            conv_layer.append(x)
            print ('x',x)
            x = conv2d_layer(x, 128, 4, 4, 2, 2, name='g_conv2')
            print ('x',x)
            x = bn_layer(x, is_training=is_training, scope='g_bn_test')
            x = lrelu_layer(x)
            # x = lrelu_layer(bn_layer(conv2d_layer(x, 128, 4, 4, 2, 2, name='g_conv2'), is_training=is_training, scope='g_bn2'))
            conv_layer.append(x)
            x = lrelu_layer(bn_layer(conv2d_layer(x, 256, 4, 4, 2, 2, name='g_conv3'), is_training=is_training, scope='g_bn3'))
            conv_layer.append(x)
            x = lrelu_layer(bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='g_conv4'), is_training=is_training, scope='g_bn4'))
            #conv_layer.append(x)
            #x = lrelu_layer(bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='g_conv5'), is_training=is_training, scope='g_bn5'))
            #conv_layer.append(x)
            #x = lrelu_layer(bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='g_conv6'), is_training=is_training, scope='g_bn6'))
            #conv_layer.append(x)g_
            #print (x.shape)
            #x = lrelu_layer(bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='g_conv7'), is_training=is_training, scope='g_bn7'))
            #conv_layer.append(x)
            #print (x.shape)
            #x = lrelu_layer(bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='g_conv8'), is_training=is_training, scope='g_bn8'))
            #print (x.shape)

            #x = lrelu_layer(bn_layer(deconv2d_layer(x, 512, 4, 4, 2, 2, name='g_dconv1'), is_training=is_training, scope='gd_bn1'))
            #print (x.shape)
            #x = tf.concat([x, conv_layer.pop()], axis=3)
            #x = lrelu_layer(bn_layer(deconv2d_layer(x, 512, 4, 4, 2, 2, name='g_dconv2'), is_training=is_training, scope='gd_bn2'))
            #x = tf.concat([x, conv_layer.pop()], axis=3)
            #x = lrelu_layer(bn_layer(deconv2d_layer(x, 512, 4, 4, 2, 2, name='g_dconv3'), is_training=is_training, scope='gd_bn3'))
            #x = tf.concat([x, conv_layer.pop()], axis=3)
            #x = lrelu_layer(bn_layer(deconv2d_layer(x, 512, 4, 4, 2, 2, name='g_dconv4'), is_training=is_training, scope='gd_bn4'))
            #x = tf.concat([x, conv_layer.pop()], axis=3)
            x = lrelu_layer(bn_layer(deconv2d_layer(x, 256, 4, 4, 2, 2, name='g_dconv5'), is_training=is_training, scope='gd_bn5'))
            x = tf.concat([x, conv_layer.pop()], axis=3)
            x = lrelu_layer(bn_layer(deconv2d_layer(x, 128, 4, 4, 2, 2, name='g_dconv6'), is_training=is_training, scope='gd_bn6'))
            x = tf.concat([x, conv_layer.pop()], axis=3)
            x = lrelu_layer(bn_layer(deconv2d_layer(x, 64, 4, 4, 2, 2, name='g_dconv7'), is_training=is_training, scope='gd_bn7'))
            x = tf.concat([x, conv_layer.pop()], axis=3)
            x = lrelu_layer(bn_layer(deconv2d_layer(x, 3, 4, 4, 2, 2, name='g_dconv8'), is_training=is_training, scope='gd_bn8'))
            x = tf.tanh(x)

            return x


class Discriminator(Model):
    def __init__(self, name="discriminator"):
        super(Discriminator, self).__init__(name)

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            x = lrelu_layer(conv2d_layer(x, 64, 4, 4, 2, 2,stddev=0.2, name='d_conv1'))
            #x, num_filters, filter_height, filter_width, stride_height, stride_width, stddev=0.2, name="conv2d"
            x = lrelu_layer(bn_layer(conv2d_layer(x, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
            x = lrelu_layer(bn_layer(conv2d_layer(x, 256, 4, 4, 2, 2, name='d_conv3'), is_training=is_training, scope='d_bn3'))
            x = lrelu_layer(bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='d_conv4'), is_training=is_training, scope='d_bn4'))
            x = lrelu_layer(bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='d_conv5'), is_training=is_training, scope='d_bn5'))
            # print('x', x)
            # x = lrelu_layer(bn_layer(conv2d_layer(x, 512, 4, 4, 2, 2, name='d_conv6'), is_training=is_training, scope='d_bn6'))
            # x = conv2d_layer(x, 1, 4, 4, 1, 1, name='d_conv7')
            x = tf.reshape(x, [batch_size, -1]) # Can use tf.reduce_mean(x, axis=[1, 2, 3])
            x = linear_layer(x, 1, scope='d_fc8')

            return x











