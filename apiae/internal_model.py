import numpy as np
import tensorflow as tf
import scipy.io
from .nets import *
from .utils import *

import numpy as np
import scipy.io
import quaternion
from .apiae_net import APIAE


class InternalModel(APIAE):
    def __init__(self,
                 R, L, K_vf, K_plan, dK, dt, n_x, n_z, n_u, ur, lr,
                 learn='a', useVF=True, isTest=False, genType='b', periodic=True, doResample=True, objective=0,
                 u_ratio=1.0, taus=1.0, tauc=1.0, taui=1.0, tauc_plan=1.0, taus_plan=1.0, obstacles=None,
                 scale_fname=''):
        self.R = R  # the number of improvements
        self.L = L  # the number of trajectory sampled
        self.K_vf = K_vf  # the number of time steps for variational flow (RNN)
        self.K_plan = K_plan  # the number of time steps
        self.dK = dK  # re-planning step interval
        self.dt = dt  # time interval
        self.sdt = np.sqrt(dt)  # sqrt(dt)
        self.B = 1  # the number of batch
        
        self.n_x = n_x  # dimension of x; observation
        self.n_z = n_z  # dimension of z; latent space
        self.n_u = n_u  # dimension of u; control

        self.ur = ur  # update rate
        self.lr = lr  # learning rate
        self.taus = taus
        self.tauc = tauc
        self.taui = taui
        self.tauc_plan = tauc_plan  # 1.0
        self.taus_plan = taus_plan  # 0.03
        self.samp_amp = tf.placeholder(tf.float32)
        self.u_ratio = u_ratio
        self.obstacles = obstacles

        self.genType = genType
        self.learn = learn  # 'a' - learn apiae, 'i' - learn iwae, 'f' - learn fivo, 'af' - learn apiae w/ resampling
        self.useVF = useVF  # flag whether this network uses variational flow or not.
        self.doResample = doResample
        self.periodic = periodic
        self.objective = objective
            
        D = np.load('./data/reference/scale_{}.npz'.format(scale_fname))
        self.x_std = tf.constant(D["x_std"], dtype=tf.float32)
        self.x_mean = tf.constant(D["x_mean"], dtype=tf.float32)
        
        # placeholders for filtering
        self.useq_tf = tf.placeholder(tf.float32, shape=(1, 1, self.K_vf-1, self.n_u))  # previous inputs
        self.useq_tf = self.useq_tf * self.u_ratio
        if self.useVF:
            self.sseq_tf = tf.placeholder(tf.float32, shape=(1, 1, self.K_vf, self.n_x))  # current observation (for past K_vf)
        
        # for planning
        self.mu_tf = tf.placeholder(tf.float32, shape=(1, 1, 1, self.n_z))  # current latent state
        self.Sig_tf = tf.placeholder(tf.float32, shape=(1, 1, 1, self.n_z))  # current latent state
        self.xglobal_obs_tf = tf.placeholder(tf.float32, shape=(1, 1, 1, 3))  # current observation
        self.theta_obs_tf = tf.placeholder(tf.float32, shape=(1, 1, 1, 1))  # current observation
        
        if self.objective == 0:
            self.xglobal_goal = tf.placeholder(tf.float32, shape=(1, 1, 1, 3))  # goal position
        elif self.objective == 1:
            self.ref_traj = tf.placeholder(tf.float32, shape=(1, 1, self.K_plan, 3))  # reference trajectory
        elif self.objective == 2:
            self.omega_target_tf = tf.placeholder(tf.float32, shape=())  # omega to follow

        # Construct networks
        self._initialize_network()
        self._create_filtering_module()
        self._create_planning_network()

        # Initializing the tensor flow variables and saver
        self.sess = tf.Session()
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess.run(init)

    def _initialize_network(self):
        # build dynamics and generative network
        self._build_networks()
        
        # initialize p0
        self.mu0 = tf.zeros((self.B, 1, 1, self.n_z))  # (B,1,1,n_z)
        self.Sig0 = tf.concat([1.0 * tf.ones((1, 1, 1, 1)), 1.0 * tf.ones((1, 1, 1, self.n_z-1))], axis=-1)  # initialzie witth arbitrary value
        self.ldet_Sig0 = tf.reduce_sum(tf.log(self.Sig0))
        
        self.museq_list = []  # debug
        self.uffseq_list = []  # debug
        self.dwseq_list = []  # debug
    
    def _create_filtering_module(self):
        # Call variational flow (q)
        if self.useVF:
            self.xseq = (self.sseq_tf - self.x_mean) / self.x_std  # copy for code re-use original APIAE code
            self._variational_flow(self.K_vf)  # Run variational flow
        else:
            self.muhat = tf.zeros((self.B, 1, 1, self.n_z))  # (B,1,1,n_z)
            self.Sighat = self.Sig0  # (B,1,1,n_z)
            self.uffhat = tf.zeros((self.B, 1, self.K_vf - 1, self.n_u))  # (B,1,K-1,n_u)

        zseq, alpha, _ = self.path_integral(self.muhat, self.Sighat, self.uffhat, self.useq_tf, self.K_vf, self.dK, isPlanner=False)
        self.zt = zseq[:,:,-1:,:]
        self.wt = alpha[:,:,-1:,:]
        self.mut = tf.reduce_sum(self.zt * self.wt, axis=1, keepdims=True)
        self.Sigt = tf.reduce_sum((self.zt - self.mut)**2 * self.wt, axis=1, keepdims=True)
        
    def _create_planning_network(self):
        zeroseq = tf.zeros((1, 1, self.K_plan-1, self.n_u))
        _, _, uffseq = self.path_integral(self.mu_tf, self.Sig_tf, zeroseq, zeroseq, self.K_plan, self.dK, isPlanner=True)
        self.uff_planned = uffseq[:, :, 0:1, :] / self.u_ratio

    def path_integral(self, muhat, Sighat, uffseq, uffseq_bias, K, dK, isPlanner):
        # Run for R times
        for r in range(self.R):
            # Sampling z0(initial latent state) and dwseq(sequence of dynamic noise)
            z0, dwseq = self.sampler(muhat, Sighat, K, dK)

            # Run dynamics and calculate the cost
            if isPlanner:
                zseq, dwseqnew, alpha, x_pos_seq, x_theta_seq = self.Simulate_Planning(z0, muhat, Sighat, uffseq, dwseq, K, dK)
            else:
                zseq, dwseqnew, alpha = self.Simulate_Filtering(z0, muhat, Sighat, uffseq, uffseq_bias, dwseq, K, dK)

            # Reshape alpha
            alpha = tf.expand_dims(tf.expand_dims(alpha, axis=-1), axis=-1)  # (B,L,1,1)

            # Update optimal control sequence & initial state dist.
            muhat, Sighat, uffseq = self.update(zseq, dwseqnew, alpha, muhat, Sighat, uffseq)
        
        if isPlanner:  # debug
            self.museq_list.append(tf.reduce_sum(alpha * zseq, axis=1, keepdims=True))
            self.uffseq_list.append(uffseq)
            self.dwseq = dwseq
            # self.x_pos_seq = tf.reduce_sum(alpha * x_pos_seq, axis=1, keepdims=True)
            # self.x_theta_seq = tf.reduce_sum(alpha * x_theta_seq, axis=1, keepdims=True)
            self.x_pos_seq = x_pos_seq
            self.x_theta_seq = x_theta_seq
            # self.dwseqnew = dwseqnew
            self.alpha = alpha
        return zseq, alpha, uffseq

    def sampler(self, muhat, Sighat, K, dK):
        # For initial states
        epsilon_z = tf.random_normal((self.B, self.L, 1, self.n_z), 0., 1., dtype=tf.float32)  # (B,L,1,n_z)
        sighat = tf.sqrt(Sighat)  # (B,1,1,n_z)
        z0 = muhat + sighat * epsilon_z  # (B,L,1,n_z)

        # For dynamic noise
        epsilon_u = tf.random_normal((self.B, self.L, int((K-2)/dK)+1, self.n_u), 0., 1.,
                                     dtype=tf.float32)  # sample noise from N(0,1)

        return z0, epsilon_u * self.sdt / np.sqrt(dK) * self.samp_amp
    
    def Simulate_Filtering(self, z0, muhat, Sighat, uffseq, uffseq_bias, dwseq, K, dK):
        # Load initial states, dynamic noise and control sequence
        dwseq_new = tf.zeros_like(dwseq[:, :, 0:1, :])  # (B,L,1,n_u)
        zk = z0  # (B,L,1,n_z)
        zseq = z0
        
        # initialize S with initial cost
        ss0 = self.state_cost(self.xseq[:, :, 0:1, :], zk)  # (B,L,1,1)
        si0 = self.initial_cost(z0, muhat, Sighat)  # (B,L,1,1)
        S = tf.squeeze(si0 + ss0, axis=[-1, -2])  # (B,L)
        log_weight = -S - tf.log(self.L * 1.0)  # (B,L)
        log_norm = tf.reduce_logsumexp(log_weight, axis=1, keepdims=True)  # (B,1)
        log_weight = log_weight - log_norm  # (B,L)

        bound = log_norm  # (B,1)

        # Compute optimal control with standardized linear feedback policy
        for k in range(K - 1):
            # Propagate
            k_dwk = int(k/dK)
            dwk = dwseq[:, :, k_dwk:k_dwk+1, :]  # (B,L,1,n_u)
            uffk = uffseq[:, :, k:k+1, :] + uffseq_bias[:,:,k:k+1,:]  # (B,1,1,n_u)
            zk = self.propagate(zseq[:, :, k:k + 1, :], uffk + dwk / self.dt)

            # Concatenate
            zseq = tf.concat([zseq, zk], axis=2)  # (B,L,k+2,n_z)
            dwseq_new = tf.concat([dwseq_new, dwk], axis=2)  # (B,L,k+2,n_u)

            # Compute control cost
            sck = self.control_cost(uffk - uffseq_bias[:,:,k:k+1,:], dwk)
            ssk = self.state_cost(self.xseq[:, :, k+1:k+2, :], zk)  # (B,L,1,1)

            # Update cost
            Sk = tf.squeeze(ssk + sck, axis=[-1, -2])  # (B,L)
            log_weight = log_weight - Sk  # (B,L)
            log_norm = tf.reduce_logsumexp(log_weight, axis=1, keepdims=True)  # (B,1)
            log_weight = log_weight - log_norm  # (B,L)

            bound = bound + log_norm
            S = S + Sk  # (B,L)

            # Resampling
            if ((k+1) % dK == 0) and self.doResample:
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
                S = tf.gather_nd(S, gather_ind)  # (B,L)
        
                should_resample_float = tf.cast(should_resample, dtype=tf.float32)
                log_weight = should_resample_float * tf.ones_like(log_weight) * (-tf.log(self.L * 1.0)) + (
                        1.0 - should_resample_float) * log_weight
                
        return zseq, dwseq_new[:, :, 1:, :], tf.exp(log_weight)
    
    def Simulate_Planning(self, z0, muhat, Sighat, uffseq, dwseq, K, dK):
        # Load initial states, dynamic noise and control sequence
        dwseq_new = tf.zeros_like(dwseq[:, :, 0:1, :])  # (B,L,1,n_u)
        zk = z0  # (B,L,1,n_z)
        zseq = z0
        
        # initialize S with initial cost
        x_pos_seq = []
        x_theta_seq = []
        x_pos = tf.tile(self.xglobal_obs_tf, (1, self.L, 1, 1))
        x_theta = tf.tile(self.theta_obs_tf, (1, self.L, 1, 1))
        x_pos_seq.append(x_pos)  # debug
        x_theta_seq.append(x_theta)  # debug
        x_global, omega = self.z2x(zk, x_pos, x_theta)  # (B,L,1,1)

        ss0 = self.planning_cost(x_pos, 0, omega)
        # si0 = 0.0 * self.initial_cost(z0, muhat, Sighat)  # (B,L,1,1)
        S = tf.squeeze(ss0, axis=[-1, -2])  # (B,L)  si0 +
        log_weight = -S - tf.log(self.L * 1.0)  # (B,L)
        log_norm = tf.reduce_logsumexp(log_weight, axis=1, keepdims=True)  # (B,1)
        log_weight = log_weight - log_norm  # (B,L)

        bound = log_norm  # (B,1)

        if self.objective == 0:
            reached = self.check_reached(x_pos)
            crashed = self.check_obstacle(x_pos)

        # Compute optimal control with standardized linear feedback policy
        for k in range(K - 1):
            # Propagate
            k_dwk = int(k/dK)
            dwk = dwseq[:, :, k_dwk:k_dwk+1, :]  # (B,L,1,n_u)
            uffk = uffseq[:, :, k:k+1, :] # (B,1,1,n_u)
            zk = self.propagate(zseq[:, :, k:k + 1, :], uffk + dwk / self.dt)

            # Concatenate
            zseq = tf.concat([zseq, zk], axis=2)  # (B,L,k+2,n_z)
            dwseq_new = tf.concat([dwseq_new, dwk], axis=2)  # (B,L,k+2,n_u)

            # Compute control cost
            sck = self.tauc_plan * self.control_cost(uffk, dwk)
            
            # sampled
            x_pos, x_theta = step_global(x_global, x_pos, x_theta, self.dt)
            x_pos_seq.append(x_pos)  # debug
            x_theta_seq.append(x_theta)  # debug
        
            # cost   # * (1.0 - tf.cast(reached, tf.float32)) * (1.0 - tf.cast(crashed, tf.float32)) \
            x_global, omega = self.z2x(zk, x_pos, x_theta)  # (B,L,1,1)
            if self.objective == 0:
                reached = tf.logical_or(reached, self.check_reached(x_pos))
                crashed = tf.logical_or(crashed, self.check_obstacle(x_pos))
                # crashed = tf.logical_and(tf.logical_not(reached), crashed)
                # ssk = self.planning_cost(x_pos, k + 1, omega)
                ssk = self.planning_cost(x_pos, k + 1, omega) \
                      - 1e4 * tf.cast(reached, tf.float32) * (1.0 - tf.cast(crashed, tf.float32)) + 1e5 * tf.cast(crashed, tf.float32)
            else:
                ssk = self.planning_cost(x_pos, k + 1, omega)
                # Update cost
            Sk = tf.squeeze(ssk + sck, axis=[-1, -2])  # (B,L)  #  ssk + sck
            log_weight = log_weight - Sk  # (B,L)
            log_norm = tf.reduce_logsumexp(log_weight, axis=1, keepdims=True)  # (B,1)
            log_weight = log_weight - log_norm  # (B,L)

            bound = bound + log_norm
            S = S + Sk  # (B,L)

            # Resampling
            if ((k+1) % dK == 0) and self.doResample:
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
                S = tf.gather_nd(S, gather_ind)  # (B,L)
                
                x_pos = tf.gather_nd(x_pos, gather_ind)
                x_theta = tf.gather_nd(x_theta, gather_ind)
                x_global = tf.gather_nd(x_global, gather_ind)

                if self.objective == 0:
                    reached = tf.gather_nd(reached, gather_ind)
                    crashed = tf.gather_nd(crashed, gather_ind)

                should_resample_float = tf.cast(should_resample, dtype=tf.float32)
                log_weight = should_resample_float * tf.ones_like(log_weight) * (-tf.log(self.L * 1.0)) \
                             + (1.0 - should_resample_float) * log_weight
                
        x_pos_seq = tf.concat(x_pos_seq, axis=2)
        x_theta_seq = tf.concat(x_theta_seq, axis=2)

        if self.objective == 0:
            self.reached_debug = tf.cast(reached, tf.float32)
            self.crashed_debug = tf.cast(crashed, tf.float32)

        return zseq, dwseq_new[:, :, 1:, :], tf.exp(log_weight), x_pos_seq, x_theta_seq
    
    def z2x(self, zt, x_pos, x_theta):
        xt_mean, xt_logSig = self.genNet.compute_x(zt)
        xt_loc = xt_mean + self.x_mean
        xt_global = loc2global(xt_loc, x_pos, x_theta)
        return xt_global, xt_loc[:, :, :, 9:10]  # (..., 1, 1)

    def rollout(self, K, z0_np, useq_np, xglobal_obs_np, theta_obs_np):
        assert useq_np.shape[2] == (K - 1), "wrong shape of useq_np"
        
        useq = tf.constant(useq_np, dtype=tf.float32)
        z0 = tf.constant(z0_np, dtype=tf.float32)
        xglobal_obs_tf = tf.constant(xglobal_obs_np, dtype=tf.float32)
        theta_obs_tf = tf.constant(theta_obs_np, dtype=tf.float32)
        
        zk = z0  # (B,L,1,n_z)
        zseq = z0
        x_pos = tf.tile(xglobal_obs_tf, (1,1,1,1))
        x_theta = tf.tile(theta_obs_tf, (1,1,1,1))
        x_pos_seq = [x_pos]  # debug
        x_theta_seq = [x_theta]  # debug
        x_global, _ = self.z2x(zk, x_pos, x_theta)  # (B,L,1,1)
        
        for k in range(K - 1):
            # Propagate
            uffk = useq[:, :, k:k+1, :]
            zk = self.propagate(zseq[:, :, k:k + 1, :], uffk)

            # Concatenate
            zseq = tf.concat([zseq, zk], axis=2)  # (B,L,k+2,n_z)
            
            # sampled
            x_pos, x_theta = step_global(x_global, x_pos, x_theta, self.dt)
            x_pos_seq.append(x_pos)  # debug
            x_theta_seq.append(x_theta)  # debug
        
            # cost
            x_global, _ = self.z2x(zk, x_pos, x_theta)  # (B,L,1,1)
        
        x_pos_seq = tf.concat(x_pos_seq, axis=2)
        x_theta_seq = tf.concat(x_theta_seq, axis=2)
        return self.sess.run([x_pos_seq, x_theta_seq])
    
    def planning_cost(self, x_pos, k, omega):
        if self.objective == 0:
            cost = self.taus_plan * tf.sqrt(tf.reduce_sum(
                (self.xglobal_goal[:, :, :, 0:1]-x_pos[:, :, :, 0:1])**2
                + (self.xglobal_goal[:, :, :, 2:3]-x_pos[:, :, :, 2:3])**2,
                axis=-1, keepdims=True)
            )
        elif self.objective == 1:
            cost = self.taus_plan*tf.reduce_sum(
                (self.ref_traj[:, :, k:k+1, 0:1]-x_pos[:, :, :, 0:1])**2
                + (self.ref_traj[:, :, k:k+1, 2:3]-x_pos[:, :, :, 2:3])**2,
                axis=[-1, -2], keepdims=True
            )
        elif self.objective == 2:
            cost = self.taus_plan*tf.reduce_sum(
                (omega - self.omega_target_tf)**2,
                axis=[-1, -2], keepdims=True
            )
        else:
            raise NotImplementedError()
        return cost

    def check_obstacle(self, x_pos,  x_margin=0.3, y_margin=0.3):
        if self.obstacles is None or len(self.obstacles) == 0:
            return False
        else:
            cond_all = False
            for obs in self.obstacles:
                x, y, wx, wy = obs
                cond_x = tf.abs(x_pos[:, :, :, 0:1] - x) < wx/2 + x_margin
                cond_y = tf.abs(x_pos[:, :, :, 2:3] - y) < wy/2 + y_margin
                cond = tf.logical_and(cond_x, cond_y)
                cond_all = tf.logical_or(cond_all, cond)
            return cond_all

    def check_reached(self, x_pos, margin=0.75):
        radius = tf.sqrt((x_pos[:, :, :, 0:1] - self.xglobal_goal[:, :, :, 0:1])**2
                         + (x_pos[:, :, :, 2:3] - self.xglobal_goal[:, :, :, 2:3])**2)
        return radius <= margin
