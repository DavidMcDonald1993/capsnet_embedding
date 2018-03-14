# import tensorflow as tf 
import keras.backend as K



def masked_crossentropy(y_true, y_pred):
    '''
    y_true shape = [None, N, 2 * num_classes]
    y_pred shape = [None, N, num_classes]
    
    '''    
    mask = y_true[:,:,:1]
    y_true = y_true[:,:,1:]
    # mask, y_true = tf.split(y_true, num_or_size_splits=2, axis=-1)
    # mask shape = [None, N, num_classes]

    # assert K.sum(mask[:,0]) > 0, "zero mask"
    # mask /= K.mean(mask)
    y_pred = K.clip(y_pred, min_value=K.epsilon(), max_value=1-K.epsilon())



    # combine neighbours and batch dimension
    # num_classes = K.shape(y_true)[-1]
    # mask = K.reshape(mask, [-1, num_classes])
    # y_pred = K.reshape(y_pred, [-1, num_classes])
    # y_true = K.reshape(y_true, [-1, num_classes])

    L = y_true * -K.log(y_pred)

    L *= mask

    return K.mean(K.mean(K.sum(L, axis=-1), axis=-1))
    

def masked_margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, 2 * num_classes]
    :return: a scalar loss value.
    """

    # mask, y_true = tf.split(y_true, num_or_size_splits=2, axis=-1)
    # mask /= K.mean(mask)
    mask = y_true[:,:,:1]
    y_true = y_true[:,:,1:]
    
    # combine neighbours and batch dimension
    # num_classes = K.shape(y_true)[-1]
    # mask = K.reshape(mask, [-1, num_classes])
    # y_pred = K.reshape(y_pred, [-1, num_classes])
    # y_true = K.reshape(y_true, [-1, num_classes])

    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
        
    L *= mask

    return K.mean(K.mean(K.sum(L, axis=-1), axis=-1))

# def euclidean_negative_sampling_loss(y_true, y_pred, num_pos=1, num_neg=5):

#     # n shape = [None, D]
#     u = y_pred[:,:1]
#     v = K.permute_dimensions(y_pred[:,1:1+num_pos], [0,2,1])
#     neg_samples = K.permute_dimensions(y_pred[:,-num_neg:], [0,2,1])
    
#     uv = K.batch_dot(u, v)
#     sig_uv = K.clip(K.sigmoid(uv), min_value=K.epsilon(), max_value=1-K.epsilon())
#     log_sig_uv = -K.log(sig_uv)
    
    
#     uneg = -K.batch_dot(u, neg_samples)
#     sig_uneg = K.clip(K.sigmoid(uneg), min_value=K.epsilon(), max_value=1-K.epsilon())
#     log_sig_uneg = -K.log(sig_uneg)
    
    
#     return K.mean(log_sig_uv + num_neg * K.mean(log_sig_uneg, axis=1))

def hyperbolic_negative_sampling_loss(y_true, y_pred):

    '''
    input shape is [None, num_pos+num_neg, D]
    y_true is a mask to allow for negative sampling over all levels 
    norm y_pred < 1
    '''

    # def sigmoid(x):
    #     return 1. / (1 + K.exp(-x))

    P = K.softmax(-y_pred)
    # P = K.softmax(-K.square(y_pred))

    # r = 1.
    # t = 1.
    # P = K.sigmoid((r - y_pred) / t)
    
    P = K.clip(P, min_value=K.epsilon(), max_value=1-K.epsilon())
    
    # return K.categorical_crossentropy(y_true, P)
    return -K.mean(K.log(P[:,0]))
    # return - K.mean( K.log(P[:,0]) + K.mean(K.log(1 - P[:,1:]), axis=1) )
