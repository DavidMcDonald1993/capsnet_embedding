import numpy as np

import tensorflow as tf 
import keras.backend as K

from tensorflow.contrib.distributions import Bernoulli


max_norm = np.nextafter(1, 0, dtype=K.floatx())
max_ = np.finfo(K.floatx()).max
min_norm = 1e-7

    
def masked_margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, 1 + n_classes]
    :param y_pred: [None, num_classes]
    :return: a scalar loss value.
    """
    mask = y_true[:,:1]
    y_true = y_true[:,1:]

    mask /= K.maximum(np.array(1., dtype=K.floatx()), K.sum(mask))

    L = y_true * K.square(K.maximum(np.array(0., dtype=K.floatx()), 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(np.array(0., dtype=K.floatx()), y_pred - 0.1))
        
    L *= mask

    return K.sum(K.sum(L, axis=-1), axis=-1)

def probabilistic_negative_sampling_loss(y_true, y_pred):

    def euclidean_distance(p, q):
        return K.sqrt(K.sum(K.square(p - q), axis=-1) + K.epsilon() )

    def kullback_leibler(p, q):
        return K.sum(p * K.log(p / q), axis=-1)

    def hellinger_distance(p, q):
        bc = K.sqrt(p * q + K.epsilon()) + K.sqrt((1 - p) * (1 - q) + K.epsilon())
        return K.sum(K.sqrt(1 - bc + K.epsilon()), axis=-1)

    y_true = K.cast(y_true, dtype=tf.int32)

    u = y_true[:,:1] 
    v = y_true[:,1:]

    u_prob = tf.nn.embedding_lookup(params=y_pred, ids=u)
    v_prob = tf.nn.embedding_lookup(params=y_pred, ids=v)

    exp_minus_dist = K.exp(-hellinger_distance(u_prob, v_prob))

    return -K.mean(K.log(exp_minus_dist[:,0]) - K.log(K.sum(exp_minus_dist[:,1:], axis=-1)))


def unsupervised_margin_loss(y_true, y_pred):



    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, 1 + n_classes]
    :param y_pred: [None, num_classes]
    :return: a scalar loss value.
    # """
    # dist = Bernoulli(probs=y_pred)
    # y_true = dist.sample()
    # y_true = K.cast(y_true, dtype=K.floatx())

    # y_true = K.one_hot(indices=K.argmax(y_pred, axis=1), 
    #     num_classes=y_pred.get_shape().as_list()[1])
    # y_true = tf.zeros_like(y_pred)

    # y_true, y_pred = tf.split(y_pred, 2, axis=-1)

    y_true = K.cast(y_true, dtype=tf.int32)

    u = y_true[:,0] 
    v = y_true[:,1]

    u_prob = tf.nn.embedding_lookup(params=y_pred, ids=u)
    v_prob = tf.nn.embedding_lookup(params=y_pred, ids=v)

    y_true = K.one_hot(indices=K.argmax(v_prob, axis=-1), num_classes=v_prob.shape[1])
    # return 

    L = y_true * K.square(K.maximum(np.array(0., dtype=K.floatx()), 0.9 - y_pred)) + \
        1.0 * (1 - y_true) * K.square(K.maximum(np.array(0., dtype=K.floatx()), y_pred - 0.1))

    return K.mean(K.sum(L, axis=-1), axis=-1)


def hyperbolic_negative_sampling_loss(y_true, y_pred):

    '''
    y_pred shape is [None, dim]
    y_true is a mask to allow for negative sampling over all levels 
    '''

    def euclidean_distance(u, v):

        def squared_norm(x, axis=-1, ):
            y = K.sum(K.square(x), axis=axis, keepdims=False)
            return y

        # return squared_norm(u - v)
        return K.sqrt(squared_norm(u - v) + K.epsilon())

    def hyperbolic_distance(u, v):

        def squared_norm(x, axis=-1, ):
            y = K.sum(K.square(x), axis=axis, keepdims=False)
            return y

        def acosh(x):
            return K.log(x + K.sqrt(K.square(x) - 1))

        num = squared_norm(u - v)
        nu = squared_norm(u)
        nv = squared_norm(v)

        num = K.clip(num, min_value=min_norm, max_value=max_)
        nu = K.clip(nu, min_value=min_norm, max_value=max_norm)
        nv = K.clip(nv, min_value=min_norm, max_value=max_norm)

        den = (1 - nu) * (1 - nv) 

        d = num / den
        d = K.clip(d, min_value=min_norm, 
            max_value=np.nextafter(np.sqrt(max_), 0, dtype=K.floatx()) / 2)

        return acosh(1 + 2 * d)

    y_true = K.cast(y_true, dtype=tf.int32)

    u = y_true[:,:1] 
    samples = y_true[:,1:]

    u_emb = tf.nn.embedding_lookup(params=y_pred, ids=u)
    sample_emb = tf.nn.embedding_lookup(params=y_pred, ids=samples)

    # distances = euclidean_distance(u_emb, sample_emb)
    distances = hyperbolic_distance(u_emb, sample_emb)

    exp_minus_d_sq = K.exp(-K.square(distances)) 
    exp_minus_d_sq = K.clip(exp_minus_d_sq, min_value=min_norm, max_value=max_norm)
    return - K.mean( K.log(exp_minus_d_sq[:,0]) - K.log( K.sum(exp_minus_d_sq[:,1:], axis=-1) ) )
