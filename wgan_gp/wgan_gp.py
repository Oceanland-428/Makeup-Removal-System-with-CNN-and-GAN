import os
import sys
import models
import tensorflow as tf
import glob
from PIL import Image

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

X, Y = create_datasets(no_dir, yes_dir)





def train_model():
	graph = tf.Graph()
    with graph.as_default():

    	def add_gradient_summary(list_gradvar):
		    # Add summary for gradients
		    for g,v in list_gradvar:
		        if g is not None:
		            tf.summary.histogram(v.name + "/gradient", g)

		def nextBatch(X, Y, num, batch_size):
            return X[num * batch_size:,:,:,:], Y[num * batch_size:,:,:,:]

		X_with = tf.placeholder(tf.float32, (batch_size, 96, 96, 3))
	    X_without = tf.placeholder(tf.float32, (batch_size, 96, 96, 3))
		G = models.Generator()
		D = models.Discriminator()

		G_opt = tf.train.AdamOptimizer(learning_rate=1E-4, name='G_opt', beta1=0.5, beta2=0.9)
	    D_opt = tf.train.AdamOptimizer(learning_rate=1E-4, name='D_opt', beta1=0.5, beta2=0.9)

	    X_fake = G(X_with)
	    X_real = X_without
	    D_real = D(X_real)
	    D_fake = D(X_fake, reuse=True)

	    G_loss = -tf.reduce_mean(D_fake)
	    D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)

	    epsilon = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
	    X_hat = X_real + epsilon * (X_fake - X_real)
	    D_X_hat = D(X_hat, reuse=True)
	    grad_D_X_hat = tf.gradients(D_X_hat, [X_hat])[0]
	    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat)))
	    gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
	    D_loss += 10 * gradient_penalty

	    dict_G_vars = G.get_trainable_variables()
	    G_vars = [dict_G_vars[k] for k in dict_G_vars.keys()]

		dict_D_vars = D.get_trainable_variables()
	    D_vars = [dict_D_vars[k] for k in dict_D_vars.keys()]

	    G_gradvar = G_opt.compute_gradients(G_loss, var_list=G_vars, colocate_gradients_with_ops=True)
	    G_update = G_opt.apply_gradients(G_gradvar, name='G_loss_minimize')

	    D_gradvar = D_opt.compute_gradients(D_loss, var_list=D_vars, colocate_gradients_with_ops=True)
	    D_update = D_opt.apply_gradients(D_gradvar, name='D_loss_minimize')

	    loss_ops = [G_loss, D_loss]

	    add_gradient_summary(G_gradvar)
	    add_gradient_summary(D_gradvar)

	    # Add scalar symmaries
	    tf.summary.scalar("G loss", G_loss)
	    tf.summary.scalar("D loss", D_loss)
	    tf.summary.scalar("gradient_penalty", gradient_penalty)

	    summary_op = tf.summary.merge_all()

	with tf.Session(graph=graph) as sess:
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        iterations = 3000
        for i in range(iterations):
            for num in range(num_batch - 1):
                with_makeup_batch, without_makeup_batch = nextBatch(X, Y, num, batch_size)
                for di in range(5):
                	sess.run([D_update], feed_dict={X_with: with_makeup_batch, X_without: without_makeup_batch})
                
                output = sess.run([G_update] + loss_ops + [summary_op], feed_dict={X_with: with_makeup_batch, X_without: without_makeup_batch})

            print ("epoch, output: ", i, output)




