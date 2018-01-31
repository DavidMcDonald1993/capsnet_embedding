'''

Much of this code is adapted from code written by 
Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
'''
import numpy as np

import keras.backend as K
import tensorflow as tf
from keras import initializers, layers, activations
from keras.regularizers import l1_l2, l2
from keras.initializers import RandomUniform



class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    
    Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
    """
    def call(self, inputs):
        return K.sqrt(K.sum(K.square(inputs), axis=-1) + K.epsilon())

    def compute_output_shape(self, input_shape):
    	# print "length output shape", input_shape[:-1]
        return input_shape[:-1]

def squash(vectors, axis=-1):
	"""
	The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
	:param vectors: some vectors to be squashed, N-dim tensor
	:param axis: the axis to squash
	:return: a Tensor with same shape as input vectors
	Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
	"""
	s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
	scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
	return scale * vectors

class GraphCapsuleLayer(layers.Layer):
	"""
	The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the 
	neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
	from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
	[None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.
	
	:param num_capsule: number of capsules in this layer
	:param dim_capsule: dimension of the output vectors of the capsules in this layer
	:param num_routing: number of iterations for the routing algorithm

	Author: David McDonald, E-mail: `dxm237@cs.bham.ac.uk`, Github: `https://github.com/DavidMcDonald1993/capsnet_embedding.git`
	"""
	def __init__(self, num_capsule, dim_capsule, num_routing=3,
				 kernel_initializer='glorot_uniform', 
				 # kernel_initializer=RandomUniform(-0.05, 0.05),
				 kernel_regularizer=l2(1e-3),
				 **kwargs):
		super(GraphCapsuleLayer, self).__init__(**kwargs)
		self.num_capsule = num_capsule
		self.dim_capsule = dim_capsule
		self.num_routing = num_routing
		self.kernel_initializer = initializers.get(kernel_initializer)
		self.kernel_regularizer = kernel_regularizer

	def build(self, input_shape):
		assert len(input_shape) >= 4, "The input Tensor should have shape=[None, N, input_num_capsule, input_dim_capsule]"
		self.input_num_capsule = input_shape[2]
		self.input_dim_capsule = input_shape[3]

		# Transform matrix
		self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
										self.dim_capsule, self.input_dim_capsule],
								 initializer=self.kernel_initializer,
								 regularizer=self.kernel_regularizer,
								 name='W')

		self.built = True

	def call(self, inputs):
		# inputs.shape=[None, N, input_num_capsule, input_dim_capsule]
		# inputs_expand.shape=[None, N, 1, input_num_capsule, input_dim_capsule]
		inputs_expand = K.expand_dims(inputs, 2)

		# Replicate num_capsule dimension to prepare being multiplied by W
		# inputs_tiled.shape=[None, N, num_capsule, input_num_capsule, input_dim_capsule]
		inputs_tiled = K.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])
		# print "tiled in shape", inputs_tiled.shape

		# Compute `inputs * W` by scanning inputs_tiled on dimension 0.
		# y.shape=[num_capsule, input_num_capsule, input_dim_capsule]
		# W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
		# Regard the first two dimensions as `batch` dimension,
		# then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
		# inputs_hat.shape = [None, N, num_capsule, input_num_capsule, dim_capsule]
		inputs_hat = K.map_fn(lambda x: 
							  K.map_fn(lambda y: K.batch_dot(y, self.W, [2, 3]), elems=x), elems=inputs_tiled)
		# print "inhat shape", inputs_hat.shape

		# raise SystemExit
		
		# Begin: Routing algorithm ---------------------------------------------------------------------#
		# In forward pass, `inputs_hat_stopped` = `inputs_hat`;
		# In backward, no gradient can flow from `inputs_hat_stopped` back to `inputs_hat`.
		inputs_hat_stopped = K.stop_gradient(inputs_hat)
		
		# The prior for coupling coefficient, initialized as zeros.
		# b.shape = [None, N, self.num_capsule, self.input_num_capsule].
		b = tf.zeros(shape=[K.shape(inputs_hat)[0], K.shape(inputs_hat)[1],
							self.num_capsule, self.input_num_capsule])

		assert self.num_routing > 0, 'The num_routing should be > 0.'
		for i in range(self.num_routing):
			# c.shape=[batch_size, N, num_capsule, input_num_capsule]
			c = tf.nn.softmax(b, dim=2)

			# At last iteration, use `inputs_hat` to compute `outputs` in order to backpropagate gradient
			if i == self.num_routing - 1:
				# c.shape =  [batch_size, N, num_capsule, input_num_capsule]
				# inputs_hat.shape=[None, N, num_capsule, input_num_capsule, dim_capsule]
				# The first three dimensions as `batch` dimension,
				# then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
				# outputs.shape=[None, num_capsule, dim_capsule]

				# outputs = squash(K.batch_dot(c, inputs_hat, [3, 3]))  
				outputs = K.batch_dot(c, inputs_hat, [3, 3])
			else:  # Otherwise, use `inputs_hat_stopped` to update `b`. No gradients flow on this path.
				outputs = squash(K.batch_dot(c, inputs_hat_stopped, [3, 3]))

				# outputs.shape =  [None, N, num_capsule, dim_capsule]
				# inputs_hat.shape=[None, N, num_capsule, input_num_capsule, dim_capsule]
				# The first two dimensions as `batch` dimension,
				# then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
				# b.shape=[batch_size, N, num_capsule, input_num_capsule]
				b += K.batch_dot(outputs, inputs_hat_stopped, [3, 4])
		# End: Routing algorithm -----------------------------------------------------------------------#

		return outputs

	def compute_output_shape(self, input_shape):
		# print "graphcaps shape", tuple(list(input_shape[:2]) + [self.num_capsule, self.dim_capsule])
		return tuple(list(input_shape[:2]) + [self.num_capsule, self.dim_capsule])

class AggregateLayer(layers.Layer):
	"""
	A layer to perform a convolutional filter on a graph with mean aggregation
	TODO
	
	Author: David McDonald, Email: `dxm237@cs.bham.ac.uk'
	"""
	def __init__(self, num_neighbours, num_filters, new_dim, mode="mean", activation=None,
				 kernel_initializer='glorot_uniform', kernel_regularizer=l2(1e-3),
				 **kwargs):
		super(AggregateLayer, self).__init__(**kwargs)

		self.num_neighbours = num_neighbours
		self.num_filters = num_filters
		self.new_dim = new_dim
		self.mode = mode
		self.activation = activation
		self.kernel_initializer = initializers.get(kernel_initializer)
		self.kernel_regularizer = kernel_regularizer
		
	def mean_weight_initializer(self, shape):
		n_dimension, Nn = shape
		W = np.zeros((n_dimension, Nn))
		
		rows = np.repeat(np.arange(n_dimension), self.num_neighbours)
		cols = np.arange(Nn)
		
		W[rows, cols] = 1. / self.num_neighbours
		# print "W", W

		return K.constant(W)

	def build(self, input_shape):
		'''
		input_shape = [batch_size, Nn, num_caps, cap_dim]
		'''

		self.n_dimension = input_shape[1] / self.num_neighbours
		# print "n dimension", self.n_dimension
		
		# bias term ?
		if self.mode == "mean":
			initializer = self.mean_weight_initializer
		self.mean_vector = self.add_weight(shape=(self.n_dimension, input_shape[1]), 
									trainable=False,  initializer=initializer,     
								   name="mean_weight_vector")

		# self.W = self.add_weight(shape=(input_shape[2] * input_shape[3], self.num_filters * self.new_dim), 
		# 							trainable=True, initializer=self.kernel_initializer, regularizer=self.kernel_regularizer,
		# 							name="W")

		# self.bias = self.add_weight(shape=(1, self.num_filters * self.new_dim),
		# 							trainable=True, initializer=self.kernel_initializer, regularizer=self.kernel_regularizer,
		# 							name="bias")

		self.built = True

	def call(self, inputs):

		'''
		input shape = [None, Nn, num_caps, cap_dim]
		'''
		
		#aggregate over neighbours
		inputs_shaped = K.reshape(inputs, shape=[K.shape(inputs)[0], K.shape(inputs)[1], -1])
		output = K.map_fn(lambda x : 
			K.dot(self.mean_vector, x), elems=inputs_shaped)
			# K.dot(self.mean_vector, K.dot(x, self.W) + self.bias), elems=inputs_shaped)
		# output = K.reshape(output, [K.shape(output)[0], K.shape(output)[1], 
		# 	self.num_filters, self.new_dim])
		output = K.reshape(output, [K.shape(output)[0], K.shape(output)[1], 
			K.shape(inputs)[2], K.shape(inputs)[3]])
		# shape is now [None, Nn+1, num_caps, cap_dim]
		
		if self.activation is not None:
			output = activations.get(self.activation)(output)
		
		return output
		
	def compute_output_shape(self, input_shape):
		'''
		input_shape = [None, Nn, num_caps, cap_dim]
		output_shape is [None, Nn+1, num_caps, cap_dim]

		'''
		
		# print "agg layer shape", tuple([input_shape[0], self.n_dimension, input_shape[2], input_shape[3]])
		return tuple([input_shape[0], self.n_dimension, input_shape[2], input_shape[3]])
		# return tuple([input_shape[0], self.n_dimension, self.num_filters, self.new_dim])

class HyperbolicDistanceLayer(layers.Layer):
    """
    TODO
    """
    def __init__(self, num_positive_samples, num_negative_samples, **kwargs):
        super(HyperbolicDistanceLayer, self).__init__(**kwargs)
        self.num_positive_samples = num_positive_samples
        self.num_negative_samples = num_negative_samples

    def build(self, input_shape):
    	self.N = input_shape[1]
    	self.step_size = self.N / (1 + self.num_positive_samples + self.num_negative_samples)
    	self.built = True
        
    def safe_norm(self, x, sqrt=False):
        x = K.sum(K.square(x), axis=-1, keepdims=False) + K.epsilon()
        if sqrt:
            x = K.sqrt(x)
        return x
        
    def call(self, inputs):
        '''
        input_shape = [None, N, D]
        '''

        # print "hypdist input shape", inputs.shape#, inputs.get_shape(), inputs.shape

        # raise SystemExit

        inputs = inputs[:,0:self.N:self.step_size]
        u = inputs[:,:1]
        v = inputs[:,1:]

        # u.set_shape([None, 1, inputs.shape[2]])
        # v.set_shape([None, self.num_positive_samples+self.num_negative_samples, inputs.shape[2]])
        # print "u shape", u.shape
        # print "v.shape", v.shape


        d = tf.acosh(1 + 2 * self.safe_norm(u - v) / 
                     ((1 - self.safe_norm(u)) * (1 - self.safe_norm(v))))
        # print "d shape", d.shape
        # d = K.squeeze(d, axis=-1)

        # print "d shape", d.shape

        return d

    def compute_output_shape(self, input_shape):
    	# print "hypdist output shape ", tuple([input_shape[0], self.num_positive_samples + self.num_negative_samples])
        return tuple([input_shape[0], self.num_positive_samples + self.num_negative_samples])