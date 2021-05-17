
import numpy as np
import tensorflow as tf


def vec2mat(vec,size):
    """
    Convert vector to matrix
    vec : (..., size[0]*size[1])
    mat : (..., size[0], size[1])
    """
    shape = tf.shape(vec)
    new_shape = tf.concat([shape[:-1], size], axis=0)
    return tf.reshape(vec, new_shape)


def lnorm(x,mu,Sig):
    """
    Compute logN(x; mu,Sig)
    x, mu, Sig : (..., d)
    """
    r = (x-mu)**2 / Sig  # (..., d)
    ldet_sig = tf.reduce_sum(tf.log(Sig), axis=-1, keepdims=True)  # (..., 1)
    return tf.reduce_sum(-0.5*r - 0.5*tf.log(2*np.pi), axis=-1, keepdims=True) \
        - 0.5*ldet_sig  # (..., 1)
