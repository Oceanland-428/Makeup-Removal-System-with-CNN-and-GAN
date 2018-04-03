import os
import glob
import numpy as np
import tensorflow as tf


def normalize_image(image):

    image = tf.cast(image, tf.float32) / 255.
    image = (image - 0.5) / 0.5
    return image

def unnormalize_image(image, name=None):

    image = (image * 0.5 + 0.5) * 255.
    image = tf.cast(image, tf.uint8, name=name)
    return image


def unnormalize_image1(image, name=None):

    #image = (image * 0.5 + 0.5) * 255.
    image = tf.cast(image, tf.uint8, name=name)
    return image


def input_data(sess):

    FLAGS = tf.app.flags.FLAGS

    list_images1 = glob.glob(os.path.join(FLAGS.Xwithout, "*.jpg"))
    list_images2 = glob.glob(os.path.join(FLAGS.Xwith, "*.jpg"))

    # Read each JPEG file

    with tf.device('/cpu:0'):

        reader1 = tf.WholeFileReader()
        filename_queue1 = tf.train.string_input_producer(list_images1)
        key1, value1 = reader1.read(filename_queue1)
        channels = FLAGS.channels
        image1 = tf.image.decode_jpeg(value1, channels=channels, name="dataset_image_without")
        image1.set_shape([None, None, channels])

        # Crop and other random augmentations
        #image1 = tf.image.random_flip_left_right(image1)
        # image = tf.image.random_saturation(image, .95, 1.05)
        # image = tf.image.random_brightness(image, .05)
        # image = tf.image.random_contrast(image, .95, 1.05)

        # Center crop
        #image1 = tf.image.central_crop(image1, FLAGS.central_fraction)

        # Resize
        image1 = tf.image.resize_images(image1, (FLAGS.img_size, FLAGS.img_size), method=tf.image.ResizeMethod.AREA)

        # Normalize
        image1 = normalize_image(image1)

        # Format image to correct ordering
        if FLAGS.data_format == "NCHW":
            image1 = tf.transpose(image1, (2,0,1))

        # Using asynchronous queues
        img_batch1 = tf.train.batch([image1, key1],
                                   batch_size=FLAGS.batch_size,
                                   num_threads=1,
                                   capacity=FLAGS.batch_size,
                                   name='X_real_input')
        
        
        reader2 = tf.WholeFileReader()
        filename_queue2 = tf.train.string_input_producer(list_images2)
        key2, value2 = reader2.read(filename_queue2)
        image2 = tf.image.decode_jpeg(value2, channels=channels, name="dataset_image_with")
        image2.set_shape([None, None, channels])

        # Crop and other random augmentations
        #image2 = tf.image.random_flip_left_right(image2)
        # image = tf.image.random_saturation(image, .95, 1.05)
        # image = tf.image.random_brightness(image, .05)
        # image = tf.image.random_contrast(image, .95, 1.05)

        # Center crop
        #image2 = tf.image.central_crop(image2, FLAGS.central_fraction)

        # Resize
        image2 = tf.image.resize_images(image2, (FLAGS.img_size, FLAGS.img_size), method=tf.image.ResizeMethod.AREA)

        # Normalize
        image2 = normalize_image(image2)

        # Format image to correct ordering
        if FLAGS.data_format == "NCHW":
            image2 = tf.transpose(image2, (2,0,1))

        # Using asynchronous queues
        img_batch2 = tf.train.batch([image2, key2],
                                   batch_size=FLAGS.batch_size,
                                   num_threads=1,
                                   capacity=FLAGS.batch_size,
                                   name='X_fake_input')
        
        
        
        #print (img_batch1.shape, img_batch2.shape)
        return img_batch1, img_batch2


def sample_batch(X, batch_size):

    idx = np.random.choice(X.shape[0], batch_size, replace=False)
    return X[idx]
