
import logging
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.python.util import deprecation

from .nets import DynNetLL, GenNet
from .utils import lnorm

deprecation._PRINT_DEPRECATION_WARNINGS = False


class APIAE(object):
    """
    APIAE for posterior inference in the latent space
    Data shape : (Batch(B), Samples(L), Time(K), Dim)
    """

    def __init__(
            self, R, L, K, dt, n_x, n_z, n_u, ur, lr, scale_fname,
            u_ratio=1.0, taus=1.0, tauc=1.0, taui=1.0,
    ):
        self.R = R  # the number of improvements
        self.L = L  # the number of trajectory sampled
        self.K = K  # the number of time steps
        self.dt = dt  # time interval
        self.sdt = np.sqrt(dt)  # sqrt(dt)

        self.n_x = n_x  # dimension of x; observation
        self.n_z = n_z  # dimension of z; latent space
        self.n_u = n_u  # dimension of u; control

        self.ur = ur  # update rate
        self.lr = lr  # learning rate
        self.taus = taus
        self.tauc = tauc
        self.taui = taui
        self.u_ratio = u_ratio

        self.xseq = tf.placeholder(tf.float32, shape=(None, 1, self.K, self.n_x))  # input sequence of observations
        self.uffseq_bias = tf.placeholder(tf.float32, shape=(None, 1, self.K - 1, self.n_u))  # input sequence of controls
        self.uffseq_bias = self.u_ratio * self.uffseq_bias
        self.B = tf.shape(self.xseq)[0]  # the number of batch
        
        scale_data = np.load(scale_fname)
        self.x_std = tf.constant(scale_data['x_std'], dtype=tf.float32)
        self.x_mean = tf.constant(scale_data['x_mean'], dtype=tf.float32)

        # construct network
        self._create_network()
        self._create_loss_optimizer()

        # initializing the tensor flow variables and saver
        self.sess = tf.Session()
        init = tf.global_variables_initializer()

        # launch the session
        self.sess.run(init)
        tf.set_random_seed(1234)

    def _create_network(self):
        # build dynamics and generative network
        self._build_networks()

        # initialize prior distribution
        self.mu0 = tf.zeros((1, 1, 1, self.n_z))
        self.sig0 = tf.ones((1, 1, 1, self.n_z))
        self.ldet_sig0 = tf.reduce_sum(tf.log(self.sig0))

        # compute posterior by using variational flow
        self._variational_flow(self.K)

        # update posterior
        self._adaptive_path_integral()

    def _build_networks(self):
        # Define the networks for dynamics and generative model
        self.dynNet = DynNetLL(self.n_z, self.n_u)
        self.genNet = GenNet(self.n_z, self.n_x, hidden_sizes=[128])

    def _variational_flow(self, K):
        self.muhat_layer = tf.layers.Dense(units=self.n_z)
        self.logsighat_layer = tf.layers.Dense(units=self.n_z)  # assume diagonal matrix
        self.uffhat_layer = tf.layers.Dense(units=self.n_u)
        self.var_layers = [self.muhat_layer, self.logsighat_layer, self.uffhat_layer]

        self.cell = tf.contrib.rnn.BasicRNNCell(num_units=8, activation=tf.nn.tanh)
        xseq_reverse = tf.reverse(tf.reshape(self.xseq, (-1, K, self.n_x)), axis=[1])
        inputs = tf.unstack(xseq_reverse, axis=1)
        hidden_list, hidden_initial = tf.nn.static_rnn(self.cell, inputs, dtype=tf.float32)
        hidden_states = tf.stack(hidden_list, axis=1)  # (B,K,n_h)
        hidden_concat = tf.reverse(
            tf.concat([hidden_states[:, :-1, :], hidden_states[:, 1:, :]], axis=2),
            axis=[1]
        )  # (B,K-1,2*n_h)

        self.muhat = tf.reshape(self.muhat_layer(hidden_initial), (-1, 1, 1, self.n_z))  # (B,1,1,n_z)
        self.sighat = tf.reshape(
            tf.exp(self.logsighat_layer(hidden_initial)) + 1e-9, (-1, 1, 1, self.n_z)
        )  # (B,1,1,n_z)
        self.uffhat = tf.reshape(self.uffhat_layer(hidden_concat), (-1, 1, K-1, self.n_u))  # (B,1,K-1,n_u)

    def _adaptive_path_integral(self):
        # initialize
        self.bound_list = []
        self.museq_list = []
        self.uffseq_list = []

        # update for R times
        muhat = self.muhat
        sighat = self.sighat
        uffseq = self.uffhat
        for r in range(self.R):
            # sampling z0(initial latent state) and dwseq(sequence of dynamic noise)
            z0, dwseq = self.sampler(muhat, sighat)

            # run dynamics and calculate the cost
            zseq, dwseqnew, alpha, bound = self.simulate(z0, muhat, sighat, uffseq, dwseq)

            # reshape alpha
            alpha = tf.expand_dims(tf.expand_dims(alpha, axis=-1), axis=-1)  # (B,L,1,1)

            # update optimal control sequence & initial state dist.
            muhat, sighat, uffseq = self.update(zseq, dwseqnew, alpha, muhat, sighat, uffseq)

            # save variables
            self.bound_list.append(bound)
            self.museq_list.append(tf.reduce_sum(alpha * zseq, axis=1, keepdims=True))
            self.uffseq_list.append(uffseq)

    def _create_loss_optimizer(self):
        self.loss = -self.bound_list[-1]
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.opt = self.optimizer.minimize(self.loss)

    def sampler(self, muhat, sighat):
        # for initial states
        epsilon_z = tf.random_normal((self.B, self.L, 1, self.n_z), 0., 1., dtype=tf.float32)  # (B,L,1,n_z)
        sighat = tf.sqrt(sighat)  # (B,1,1,n_z)
        z0 = muhat + sighat * epsilon_z  # (B,L,1,n_z)

        # for dynamic noise
        epsilon_u = tf.random_normal(
            (self.B, self.L, self.K - 1, self.n_u), 0., 1.,
            dtype=tf.float32
        )
        return z0, epsilon_u * self.sdt

    def simulate(self, z0, muhat, sighat, uffseq, dwseq):
        # load initial states, dynamic noise and control sequence
        dwseq_new = tf.zeros_like(dwseq[:, :, 0:1, :])  # (B,L,1,n_u)
        zk = z0  # (B,L,1,n_z)
        zseq = z0
        
        # initialize cost with initial cost
        ss0 = self.state_cost(self.xseq[:, :, 0:1, :], zk)  # (B,L,1,1)
        si0 = self.initial_cost(z0, muhat, sighat)  # (B,L,1,1)
        s0 = tf.squeeze(si0 + ss0, axis=[-1, -2])  # (B,L)
        log_weight = -s0 - tf.log(self.L * 1.0)  # (B,L)
        log_norm = tf.reduce_logsumexp(log_weight, axis=1, keepdims=True)  # (B,1)
        log_weight = log_weight - log_norm  # (B,L)

        bound = log_norm  # (B,1)

        # compute optimal control with standardized linear feedback policy
        s = s0
        for k in range(self.K - 1):
            # propagate
            dwk = dwseq[:, :, k:k + 1, :]  # (B,L,1,n_u)
            uffk = uffseq[:, :, k:k + 1, :]  # (B,1,1,n_u)
            zk = self.propagate(zseq[:, :, k:k + 1, :], uffk + dwk / self.dt)

            # concatenate
            zseq = tf.concat([zseq, zk], axis=2)  # (B,L,k+2,n_z)
            dwseq_new = tf.concat([dwseq_new, dwk], axis=2)  # (B,L,k+2,n_u)

            # compute control cost
            sck = self.control_cost(uffk - self.uffseq_bias[:, :, k:k+1, :], dwk)

            # compute state cost
            ssk = self.state_cost(self.xseq[:, :, k + 1:k + 2, :], zk)  # (B,L,1,1)
        
            # update cost
            sk = tf.squeeze(ssk + sck, axis=[-1, -2])  # (B,L)
            log_weight = log_weight - sk  # (B,L)
            log_norm = tf.reduce_logsumexp(log_weight, axis=1, keepdims=True)  # (B,1)
            log_weight = log_weight - log_norm  # (B,L)

            bound = bound + log_norm
            s += sk  # (B,L)
            
            # resampling
            ess = 1 / tf.reduce_sum(tf.exp(2 * log_weight), axis=1, keepdims=True)  # (B,1)
            should_resample = tf.cast(ess < 1.0 * self.L, dtype=tf.int32)  # (B,1)
            oind = np.expand_dims(np.arange(self.L), 0)  # (1,L)

            dist = tf.distributions.Categorical(logits=log_weight)
            rind = tf.stop_gradient(tf.transpose(dist.sample(self.L)))  # (B,L)

            new_ind = should_resample * rind + (1 - should_resample) * oind  # (B,L)
            bat_ind = tf.tile(tf.expand_dims(tf.range(tf.shape(new_ind)[0]), -1), (1, self.L))  # (B,L)
            gather_ind = tf.stack([bat_ind, new_ind], axis=-1)  # (B,L,2)

            zseq = tf.gather_nd(zseq, gather_ind)  # (B,L,k+2,n_z)
            dwseq_new = tf.gather_nd(dwseq_new, gather_ind)  # (B,L,k+2,n_u)
            s = tf.gather_nd(s, gather_ind)  # (B,L)
            should_resample_float = tf.cast(should_resample, dtype=tf.float32)
            log_weight = should_resample_float * tf.ones_like(log_weight) * (-tf.log(self.L * 1.0)) \
                + (1.0 - should_resample_float) * log_weight
        return zseq, dwseq_new[:, :, 1:, :], tf.exp(log_weight), tf.reduce_sum(bound)

    def propagate(self, zt, ut):
        """Simulate one-step forward"""
        # Input: zt=(...,n_z), ut=(...,n_u)
        # Output: znext=(...,n_z)
        at, bt, sigmat = self.dynNet.compute_ab_sigma(zt)  # (...,n_z,n_z), (...,n_z,1), (...,n_z,n_u)
        zdott = at @ tf.expand_dims(zt, axis=-1) + bt  # (...,n_z,1)
        znext = zt + tf.squeeze(zdott + sigmat @ tf.expand_dims(ut, axis=-1), axis=-1) * self.dt  # (...,n_z)
        return znext

    def initial_cost(self, z0, muhat, sighat):
        """Compute the cost of initial state"""
        q0 = lnorm(z0, muhat, sighat)
        p0 = 0.5 * tf.reduce_sum(
            -(z0 - self.mu0) ** 2 / self.sig0 - 0.5 * tf.log(2 * np.pi),
            axis=-1, keepdims=True
        ) - 0.5 * self.ldet_sig0
        return self.taui * (q0 - p0)  # (B,RL,1,1)

    def control_cost(self, uff, dw):
        """Compute the cost of control input"""
        uTu = tf.reduce_sum(uff ** 2, axis=-1, keepdims=True)  # (B,L,1,1)
        uTdw = tf.reduce_sum(uff * dw, axis=-1, keepdims=True)  # (B,L,1,1)
        return self.tauc * (0.5 * uTu * self.dt + uTdw)  # (B,L,1,1)

    def state_cost(self, xt_true, zt):  # shape of inputs: (..., 1, n_x), (..., 1, n_z)
        """Compute the log-likelihood of observation xt given latent zt"""
        xt_mean, xt_logsig = self.genNet.compute_x(zt)
        xt_sig = tf.exp(xt_logsig)
        xt_true = xt_true * self.x_std
        xt_sig = xt_sig * self.x_std**2
        cost = -lnorm(xt_true, xt_mean, xt_sig)  # (..., 1, 1)
        return self.taus * tf.reduce_sum(cost, axis=[-2, -1], keepdims=True)  # (..., 1, 1)

    def update(self, zseq, dwseq, alpha, muhat, sighat, uffseq):
        """Compute optimal control policies"""
        muhat_star = (1 - self.ur) * muhat \
            + self.ur * tf.reduce_sum(alpha * zseq[:, :, :1, :], axis=1, keepdims=True)
        sighat_star = sighat
        uffseq_star = uffseq \
            + self.ur * tf.reduce_sum(alpha * dwseq, axis=1, keepdims=True) / self.dt  # (B,1,K-1,n_u)
        return muhat_star, sighat_star, uffseq_star

    def partial_fit(self, xseq, uffseq_bias, return_value=False):
        """Train model based on mini-batch of input data, and return the cost of mini-batch."""            
        if return_value:
            feed = {self.xseq: xseq, self.uffseq_bias: uffseq_bias}
            _, loss, museq, uffseq, xseq = self.sess.run(
                (self.opt, self.loss, self.museq_list[-1], self.uffseq_list[-1],
                 self.genNet.compute_x(self.museq_list[-1])[0]),
                feed_dict=feed
            )
        else:
            feed = {self.xseq: xseq, self.uffseq_bias: uffseq_bias}
            _, loss = self.sess.run(
                (self.opt, self.loss), feed_dict=feed
            )
            museq = uffseq = xseq = 0.0
        return loss, museq, uffseq / self.u_ratio, xseq
    
    def save_weights(self, filename="weights.pkl"):
        """Save the weights of neural networks"""
        weights = {}
        for i, layer in enumerate(self.dynNet.layers):
            weights['d_w' + str(i)] = self.sess.run(layer.weights)
        for i, layer in enumerate(self.var_layers):
            weights['v_w' + str(i)] = self.sess.run(layer.weights)
        weights['v_rnn'] = self.sess.run(self.cell.weights)

        filehandler = open(filename, "wb")
        pickle.dump(weights, filehandler)
        filehandler.close()

        logging.info('weight saved in ' + filename)
        return weights

    def restore_weights(self, filename="weights.pkl"):
        """Load the weights of neural networks"""
        filehandler = open(filename, "rb")
        weights = pickle.load(filehandler)
        filehandler.close()

        for i, layer in enumerate(self.dynNet.layers):
            for j, w in enumerate(layer.weights):
                self.sess.run(tf.assign(w, weights['d_w' + str(i)][j]))
        for i, layer in enumerate(self.var_layers):
            for j, w in enumerate(layer.weights):
                self.sess.run(tf.assign(w, weights['v_w' + str(i)][j]))
        for j, w in enumerate(self.cell.weights):
            self.sess.run(tf.assign(w, weights['v_rnn'][j]))

        logging.info('weight restored from ' + filename)
        return weights
