import numpy as np

import tensorflow as tf 
import keras.backend as K


max_norm = np.nextafter(1, 0, dtype=K.floatx())
max_ = np.finfo(K.floatx()).max
min_norm = 1e-7


# def masked_crossentropy(y_true, y_pred):
#     '''
#     y_true shape = [None, N, 1 + num_classes]
#     y_pred shape = [None, N, num_classes]
    
#     '''    

#     y_true = K.reshape(y_true, shape=tf.stack([-1, K.shape(y_true)[-1]]))
#     y_pred = K.reshape(y_pred, shape=tf.stack([-1, K.shape(y_pred)[-1]]))

#     mask = y_true[:,:1]
#     y_true = y_true[:,1:]
#     y_pred = K.clip(y_pred, min_value=K.epsilon(), max_value=1-K.epsilon())

#     mask /= K.maximum(np.array(1., dtype=K.floatx()), K.sum(mask))


#     # combine neighbours and batch dimension
#     L = y_true * -K.log(y_pred)

#     L *= mask

#     return K.sum(K.sum(L, axis=-1), axis=-1)
    
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

# def masked_margin_loss(y_true, y_pred):
#     """
#     Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
#     :param y_true: [None, n_classes]
#     :param y_pred: [None, 2 * num_classes]
#     :return: a scalar loss value.
#     """

#     y_true = K.reshape(y_true, shape=tf.stack([-1, K.shape(y_true)[-1]]))
#     y_pred = K.reshape(y_pred, shape=tf.stack([-1, K.shape(y_pred)[-1]]))

#     mask = y_true[:,:1]
#     y_true = y_true[:,1:]

#     mask /= K.maximum(np.array(1., dtype=K.floatx()), K.sum(mask))

#     L = y_true * K.square(K.maximum(np.array(0., dtype=K.floatx()), 0.9 - y_pred)) + \
#         0.5 * (1 - y_true) * K.square(K.maximum(np.array(0., dtype=K.floatx()), y_pred - 0.1))
        
#     L *= mask

#     return K.sum(K.sum(L, axis=-1), axis=-1)

def prediction_hyperbolic_negative_sampling_loss(y_true, y_pred):
    pass

def hyperbolic_negative_sampling_loss(y_true, y_pred):

    '''
    y_pred shape is [None, dim]
    y_true is a mask to allow for negative sampling over all levels 
    '''

    def hyperbolic_distance(u, v):

        def squared_norm(x, axis=-1, sqrt=False, clip=False):
            y = K.sum(K.square(x), axis=axis, keepdims=False)
            return y

        def acosh(x):
            return K.log(x + K.sqrt(K.square(x) - 1))

        num = squared_norm(u - v)
        nu = squared_norm(u)
        nv = squared_norm(v)

        num = tf.clip_by_value(num, min_norm, max_)
        nu = tf.clip_by_value(nu, min_norm, max_norm)
        nv = tf.clip_by_value(nv, min_norm, max_norm)

        den = (1 - nu) * (1 - nv) 

        d = num / den
        d = tf.clip_by_value(d, min_norm, 
            np.nextafter(np.sqrt(max_), 0, dtype=K.floatx()) / 2)

        return acosh(1 + 2 * d)

    # print y_true.shape
    # print y_true
    y_true = K.cast(y_true, dtype=tf.int32)

    u = y_true[:,:1] 
    # pos_samples = y_true[:,1:2]
    # neg_samples = y_true[:,2:]
    samples = y_true[:,1:]

    u_emb = tf.nn.embedding_lookup(params=y_pred, ids=u)
    sample_emb = tf.nn.embedding_lookup(params=y_pred, ids=samples)

    hyperbolic_distances = hyperbolic_distance(u_emb, sample_emb)

    exp_minus_d = K.exp(-K.square(hyperbolic_distances)) 
    # exp_minus_d = K.clip(exp_minus_d, min_value=min_norm, max_value=max_norm)
    return - K.mean( K.log(exp_minus_d[:,0]) - K.log(K.sum(exp_minus_d, axis=-1) ) )
