import numpy as np

import tensorflow as tf 
import keras.backend as K



def masked_crossentropy(y_true, y_pred):
    '''
    y_true shape = [None, N, 1 + num_classes]
    y_pred shape = [None, N, num_classes]
    
    '''    

    y_true = K.reshape(y_true, shape=tf.stack([-1, K.shape(y_true)[-1]]))
    y_pred = K.reshape(y_pred, shape=tf.stack([-1, K.shape(y_pred)[-1]]))

    mask = y_true[:,:1]
    y_true = y_true[:,1:]
    y_pred = K.clip(y_pred, min_value=K.epsilon(), max_value=1-K.epsilon())

    mask /= K.maximum(np.array(1., dtype=K.floatx()), K.sum(mask))


    # combine neighbours and batch dimension
    L = y_true * -K.log(y_pred)

    L *= mask

    return K.sum(K.sum(L, axis=-1), axis=-1)
    

def masked_margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, 2 * num_classes]
    :return: a scalar loss value.
    """

    y_true = K.reshape(y_true, shape=tf.stack([-1, K.shape(y_true)[-1]]))
    y_pred = K.reshape(y_pred, shape=tf.stack([-1, K.shape(y_pred)[-1]]))

    mask = y_true[:,:1]
    y_true = y_true[:,1:]

    mask /= K.maximum(np.array(1., dtype=K.floatx()), K.sum(mask))

    L = y_true * K.square(K.maximum(np.array(0., dtype=K.floatx()), 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(np.array(0., dtype=K.floatx()), y_pred - 0.1))
        
    L *= mask

    return K.sum(K.sum(L, axis=-1), axis=-1)

def hyperbolic_negative_sampling_loss(y_true, y_pred):

    '''
    input shape is [None, 1+num_neg]
    y_true is a mask to allow for negative sampling over all levels 
    '''

    exp_minus_d = K.exp(-K.square(y_pred)) 
    # return -K.mean( exp_minus_d[:,0] - K.mean(exp_minus_d[:,1:], axis=-1) )
    # exp_minus_d = K.exp(-y_pred)
    exp_minus_d = K.clip(exp_minus_d, min_value=1e-7, max_value=np.nextafter(1, 0, dtype=K.floatx()))
    return - K.mean( K.log(exp_minus_d[:,0]) - K.log(K.sum(exp_minus_d[:,0:], axis=-1) ) )

    # P = K.softmax(-y_pred)
    # P = K.softmax(-K.square(y_pred))
    # P = K.clip(P, min_value=0.01, max_value=0.99)
    
    # return K.categorical_crossentropy(y_true, P)
    # return -K.mean(K.log(P[:,0]))
