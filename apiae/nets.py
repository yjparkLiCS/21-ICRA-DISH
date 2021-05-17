
import logging

import numpy as np
import tensorflow as tf

from .utils import vec2mat


class DynNetLL(object):
    """ Neural-net for the locally linear dynamics model: zdot=dz/dt=f(z) """
    def __init__(self, n_z, n_u, num_models=16):
        self.n_z = n_z
        self.n_u = n_u
        self.num_models = num_models
        self.output_size = n_z * n_z + n_z + n_u * n_z

        # Construct the neural network
        self.layers = []
        self.layers.append(
            tf.layers.Dense(units=self.num_models, activation=tf.nn.softmax)
        )
        self.layers.append(
            tf.layers.Dense(
                units=self.output_size, use_bias=False, name='DynLayerLast',
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1)
            )
        )
        
        # initialization
        self.z_in = tf.placeholder(tf.float32, (None, n_z))
        self.a_init = tf.placeholder(tf.float32, (n_z, n_z))
        self.b_init = tf.placeholder(tf.float32, (n_z, 1))
        self.sigma_init = tf.placeholder(tf.float32, (n_z, n_u))

        a_out, b_out, sigma_out = self.compute_ab_sigma(self.z_in)
        self.loss_A = tf.reduce_mean(
            tf.reduce_sum((a_out - tf.expand_dims(self.a_init, axis=0)) ** 2, axis=-1)
        )
        self.loss_b = tf.reduce_mean(
            tf.reduce_sum((b_out - tf.expand_dims(self.b_init, axis=0)) ** 2, axis=-1)
        )
        self.loss_sigma = tf.reduce_mean(
            tf.reduce_sum((sigma_out - tf.expand_dims(self.sigma_init, axis=0)) ** 2, axis=-1)
        )
        self.loss = self.loss_A + self.loss_b + self.loss_sigma
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def compute_ab_sigma(self, z):
        # Input :  z=(...,n_z)
        # Output: zdot=(...,n_z), sigma=(...,n_z,n_u)
        for layer in self.layers:
            z = layer(z)
        a, b, sigma = tf.split(z, [self.n_z*self.n_z, self.n_z, self.n_z*self.n_u], axis=-1)
        a_out = vec2mat(a, (self.n_z, self.n_z))
        b_out = vec2mat(b, (self.n_z, 1))
        sigma_out = vec2mat(sigma, (self.n_z, self.n_u))
        return a_out, b_out, sigma_out

    def initialize(
            self, sess, z_ref, a_init, b_init, sigma_init,
            batch_size=500, training_epochs=500, display_step=100
    ):
        n_data = z_ref.shape[0]
        total_batch = int(n_data / batch_size)

        for epoch in range(training_epochs):
            avg_loss = 0
            nperm = np.random.permutation(n_data)

            # loop over all batches
            for i in range(total_batch):
                minibatch_idx = nperm[i * batch_size:(i + 1) * batch_size]
                z_in = z_ref[minibatch_idx, :]
                feed_dict = {
                    self.z_in: z_in, self.a_init: a_init,
                    self.b_init: b_init, self.sigma_init: sigma_init
                }
                opt, loss = sess.run((self.optimizer, self.loss), feed_dict=feed_dict)
                avg_loss += loss/total_batch

            if epoch % display_step == 0:
                logging.info('Epoch={:04d}, loss={:.9f}'.format(epoch, avg_loss))


class GenNet(object):
    """ Neural-net for the generative model: x=g(z) """
    def __init__(self, n_z, n_x, hidden_sizes):
        self.hidden_sizes = hidden_sizes
        self.output_size = n_x * 2

        # construct the neural network
        self.layers = []
        for i, unit in enumerate(self.hidden_sizes):
            self.layers.append(
                tf.layers.Dense(
                    units=unit, activation=tf.nn.tanh, name='GenLayer' + str(i)
                )
            )
        self.layers.append(
            tf.layers.Dense(units=self.output_size, name='GenLayerLast')
        )

        # Below is for the later use of this network
        self.z_in = tf.placeholder(tf.float32, (None, n_z))
        
    def compute_x(self, z):
        # Input : z=(...,n_z)
        # Output: x=(...,n_x)
        for layer in self.layers:
            z = layer(z)
        x, log_sig = tf.split(z, 2, axis=-1)
        return x, log_sig
