import os
import sys
import models
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
sys.path.append("../utils")
import visualization_utils as vu
import training_utils as tu
import data_utils as du

FLAGS = tf.app.flags.FLAGS

batch_size = 16
no_dir = '/home/ubuntu/makeup_removal/gan_makeup_data_96/without/'
yes_dir = '/home/ubuntu/makeup_removal/gan_makeup_data_96/with/'

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
        if len(img_no.shape) == 3 and len(img_yes.shape) == 3 and img_no.shape[2] == 3 and img_yes.shape[2] == 3:
            X.append(np.array(img_yes))
            Y.append(np.array(img_no))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def nextBatch(X, Y, num, batch_size):
    return X[num * batch_size:(num + 1) * batch_size,:,:,:], Y[num * batch_size:(num + 1) * batch_size,:,:,:]

def train_model():

    # Setup session
    sess = tu.setup_session()

    # Setup async input queue of real images
    #X_input = du.input_data(sess)
    #X_real, X_fake_in = X_input[0], X_input[1]
    X_real, X_fake_in = du.input_data(sess)
    X_real_name = X_real[1]
    X_real = X_real[0]
    X_fake_name = X_fake_in[1]
    X_fake_in = X_fake_in[0]
    #X, Y = create_datasets(no_dir, yes_dir)
    #X_fake_in = tf.placeholder(tf.float32, (batch_size, 96, 96, 3))
    #X_real = tf.placeholder(tf.float32, (batch_size, 96, 96, 3))

    #######################
    # Instantiate generator
    #######################
    list_filters = [256, 128, 64, 3]
    list_strides = [2] * len(list_filters)
    list_kernel_size = [3] * len(list_filters)
    list_padding = ["SAME"] * len(list_filters)
    output_shape = X_real.get_shape().as_list()[1:]
    G = models.Generator(list_filters, list_kernel_size, list_strides, list_padding, output_shape,
                         batch_size=FLAGS.batch_size, data_format=FLAGS.data_format)

    ###########################
    # Instantiate discriminator
    ###########################
    list_filters = [32, 64, 128, 256]
    list_strides = [2] * len(list_filters)
    list_kernel_size = [3] * len(list_filters)
    list_padding = ["SAME"] * len(list_filters)
    D = models.Discriminator(list_filters, list_kernel_size, list_strides, list_padding,
                             FLAGS.batch_size, data_format=FLAGS.data_format)

    ###########################
    # Instantiate optimizers
    ###########################
    G_opt = tf.train.AdamOptimizer(learning_rate=1E-4, name='G_opt', beta1=0.5, beta2=0.9)
    D_opt = tf.train.AdamOptimizer(learning_rate=1E-4, name='D_opt', beta1=0.5, beta2=0.9)

    ###########################
    # Instantiate model outputs
    ###########################

    # noise_input = tf.random_normal((FLAGS.batch_size, FLAGS.noise_dim,), stddev=0.1)
    noise_input = tf.random_uniform((FLAGS.batch_size, FLAGS.noise_dim,), minval=-1, maxval=1)
    X_fake = G(X_fake_in)
    #X_fake = G(noise_input)

    # output images
    #X_G_input = du.unnormalize_image(X_fake_in)
    X_G_output = du.unnormalize_image(X_fake)
    X_real_output = du.unnormalize_image(X_real)

    D_real = D(X_real)
    D_fake = D(X_fake, reuse=True)

    ###########################
    # Instantiate losses
    ###########################

    G_loss = -tf.reduce_mean(D_fake) + (tf.reduce_mean(abs(X_fake - X_real)))
    D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)

    epsilon = tf.random_uniform(
        shape=[FLAGS.batch_size, 1, 1, 1],
        minval=0.,
        maxval=1.
    )
    X_hat = X_real + epsilon * (X_fake - X_real)
    D_X_hat = D(X_hat, reuse=True)
    grad_D_X_hat = tf.gradients(D_X_hat, [X_hat])[0]
    if FLAGS.data_format == "NCHW":
        red_idx = [1]
    else:
        red_idx = [-1]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat), reduction_indices=red_idx))
    gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
    D_loss += 10 * gradient_penalty

    ###########################
    # Compute gradient updates
    ###########################

    dict_G_vars = G.get_trainable_variables()
    G_vars = [dict_G_vars[k] for k in dict_G_vars.keys()]

    dict_D_vars = D.get_trainable_variables()
    D_vars = [dict_D_vars[k] for k in dict_D_vars.keys()]

    G_gradvar = G_opt.compute_gradients(G_loss, var_list=G_vars, colocate_gradients_with_ops=True)
    G_update = G_opt.apply_gradients(G_gradvar, name='G_loss_minimize')

    D_gradvar = D_opt.compute_gradients(D_loss, var_list=D_vars, colocate_gradients_with_ops=True)
    D_update = D_opt.apply_gradients(D_gradvar, name='D_loss_minimize')

    ##########################
    # Group training ops
    ##########################
    loss_ops = [G_loss, D_loss]

    ##########################
    # Summary ops
    ##########################

    # Add summary for gradients
    tu.add_gradient_summary(G_gradvar)
    tu.add_gradient_summary(D_gradvar)

    # Add scalar symmaries
    tf.summary.scalar("G loss", G_loss)
    tf.summary.scalar("D loss", D_loss)
    tf.summary.scalar("gradient_penalty", gradient_penalty)

    summary_op = tf.summary.merge_all()

    ############################
    # Start training
    ############################

    # Initialize session
    saver = tu.initialize_session(sess)

    # Start queues
    tu.manage_queues(sess)

    # Summaries
    writer = tu.manage_summaries(sess)
    txtfile = open("/home/ubuntu/makeup_removal/WGAN-GP/testfile.txt","w") 
    g_loss_list = []
    d_loss_list = []
    for e in tqdm(range(FLAGS.nb_epoch), desc="Training progress"):

        t = tqdm(range(FLAGS.nb_batch_per_epoch), desc="Epoch %i" % e, mininterval=0.5)
        num = 0 
        for batch_counter in t:
            #with_makeup_batch, without_makeup_batch = nextBatch(X, Y, num, batch_size)
            num += 1
            g_loss_total = 0
            d_loss_total = 0
            for di in range(5):
                sess.run([D_update])
            #    #output = sess.run([G_update] + loss_ops + [summary_op])
            
            #sess.run([D_update])
            output = sess.run([G_update] + loss_ops + [summary_op])
            g_loss, d_loss = sess.run([G_loss, D_loss])
            g_loss_total += g_loss
            d_loss_total += d_loss
            #print (g_loss, d_loss)

            if batch_counter % (FLAGS.nb_batch_per_epoch // 20) == 0:
                writer.add_summary(output[-1], e * FLAGS.nb_batch_per_epoch + batch_counter)

            t.set_description('Epoch %i' % e)
        g_loss_list.append(g_loss_total)
        d_loss_list.append(d_loss_total)
        # Plot some generated images
        #output = sess.run([X_G_output, X_G_input, X_real_output])
        output = sess.run([X_G_output, X_real_output, X_fake_name, X_real_name])
        vu.save_image(output[:2], FLAGS.data_format, e)
        name = output[2:]
        print(name)
        with open('/home/ubuntu/makeup_removal/WGAN-GP/testfile.txt', 'a') as f:
            f.write(str(e))
            f.write('\n')
            for array in name:
                for n in array:
                    f.write(str(n))
                    f.write('\n')
            f.write('\n\n')
        #txtfile.close()
        #print (name)

        # Save session
        saver.save(sess, os.path.join(FLAGS.model_dir, "model"), global_step=e)

    print('Finished training!')
    plt.plot(range(FLAGS.nb_epoch), g_loss_list, label='g_loss')
    plt.plot(range(FLAGS.nb_epoch), d_loss_list, label='d_loss')
    plt.legend()
    plt.title('G:1E-4, D:1E-4')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('/home/ubuntu/makeup_removal/WGAN-GP/G_4_D_4.png')
    #plt.show()
