'''

Much of this code is adapted from code written by 
Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
'''
# import numpy as np

import keras.backend as K
import tensorflow as tf
from keras import initializers, layers, activations#, regularizers
from keras.regularizers import l2
# from keras.initializers import RandomUniform



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
	scale = s_squared_norm / ((1 + s_squared_norm) * K.sqrt(s_squared_norm + K.epsilon()))
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
				 kernel_regularizer=1e-3,
				 **kwargs):
		super(GraphCapsuleLayer, self).__init__(**kwargs)
		self.num_capsule = num_capsule
		self.dim_capsule = dim_capsule
		self.num_routing = num_routing
		self.kernel_initializer = kernel_initializer
		self.kernel_regularizer = kernel_regularizer

	def build(self, input_shape):
		assert len(input_shape) >= 4, "The input Tensor should have shape=[None, N, input_num_capsule, input_dim_capsule]"
		# self.batch_size = input_shape[0]
		self.neighbours = input_shape[1]
		self.input_num_capsule = input_shape[2]
		self.input_dim_capsule = input_shape[3]
		
		initializer = initializers.get(self.kernel_initializer)
		# regularizer = regularizers.get(self.kernel_regularizer)
		regularizer = l2(self.kernel_regularizer)

		# Transform matrix
		# self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
		# 								self.dim_capsule, self.input_dim_capsule],
		# 						 initializer=initializer,
		# 						 regularizer=regularizer,
		# 						 name='W')
		self.W = self.add_weight(shape=[self.input_num_capsule, self.input_dim_capsule,
										self.num_capsule * self.dim_capsule],
								 initializer=initializer,
								 regularizer=regularizer,
								 name='W')

		self.built = True

	def get_config(self):
		config = super(GraphCapsuleLayer, self).get_config()
		config.update({"num_capsule":self.num_capsule,
			"dim_capsule":self.dim_capsule, "num_routing":self.num_routing,
			"kernel_initializer":self.kernel_initializer, "kernel_regularizer":self.kernel_regularizer})
		return config

	def call(self, inputs):
		# inputs.shape=[None, N, input_num_capsule, input_dim_capsule]
		# inputs_expand.shape=[None, N, 1, input_num_capsule, input_dim_capsule]
		# inputs_expand = K.expand_dims(inputs, 2)

		# Replicate num_capsule dimension to prepare being multiplied by W
		# inputs_tiled.shape=[None, N, num_capsule, input_num_capsule, input_dim_capsule]
		# inputs_tiled = K.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])

		# Compute `inputs * W` by scanning inputs_tiled on dimension 0.
		# y.shape=[num_capsule, input_num_capsule, input_dim_capsule]
		# W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
		# Regard the first two dimensions as `batch` dimension,
		# then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
		# inputs_hat.shape = [None, N, num_capsule, input_num_capsule, dim_capsule]
		# inputs_hat = K.map_fn(lambda x: 
		# 					  K.map_fn(lambda y: K.batch_dot(y, self.W, [2, 3]), elems=x), elems=inputs_tiled)

		# inputs_hat = K.map_fn(lambda x: 
		# 	# K.map_fn(lambda y: 
		# 	K.reshape(K.batch_dot(x, self.W, axes=[2, 1]), 
		# 		shape=[-1, self.input_num_capsule, self.num_capsule, self.dim_capsule]), 
		# 	# elems=x),
		# 	elems=inputs)
		# inputs_hat = K.map_fn(lambda x, shape=[-1, self.num_capsule, self.dim_capsule]: 
		# 	K.map_fn(lambda y, shape=shape: 
		# 	K.reshape(y, shape=shape), elems=x), 
		# 	elems=inputs_hat)


		# inputs shape is [None, N, input_num_caps, input_cap_dim]
		batch_size = K.shape(inputs)[0]
		inputs = K.reshape(inputs, [-1, self.input_num_capsule, self.input_dim_capsule])
		# shape is now [None*N, input num caps, input cap dim]
		inputs = K.permute_dimensions(inputs, [1, 0, 2])
		# shape is now [input num caps, None*N, input cap dim]
		# W shape is [input num caps, input cap dim, num_caps * cap dim]
		inputs_hat = K.batch_dot(inputs, self.W, axes=[2, 1])
		# shape is now [input num caps, None*N, num caps * cap dim]
		inputs_hat = K.permute_dimensions(inputs_hat, [1, 0, 2])
		# shape is now [None*N, input num caps, num_caps * caps dim]
		inputs_hat = K.reshape(inputs_hat, tf.stack([batch_size, -1, self.input_num_capsule, 
													self.num_capsule, self.dim_capsule]))

		# shape is now [None, N, input num caps, num caps, cap dim]

		inputs_hat = K.permute_dimensions(inputs_hat, pattern=[0,1,3,2,4])
		# shape is now [None, N, num_caps, num_input_caps, cap_dim]
		# print "inputs_hat", inputs_hat.shape
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
				# outputs.shape=[None, N, num_capsule, dim_capsule]

				# outputs = squash(K.batch_dot(c, inputs_hat, [3, 3]))  
				outputs = K.batch_dot(c, inputs_hat, axes=3)
			else:  # Otherwise, use `inputs_hat_stopped` to update `b`. No gradients flow on this path.
				outputs = squash(K.batch_dot(c, inputs_hat_stopped, axes=3))
				# outputs.shape =  [None, N, num_capsule, dim_capsule]
				# inputs_hat.shape=[None, N, num_capsule, input_num_capsule, dim_capsule]
				# The first two dimensions as `batch` dimension,
				# then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
				# b.shape=[batch_size, N, num_capsule, input_num_capsule]
				b += K.batch_dot(outputs, inputs_hat_stopped, axes=[3,4])
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
	def __init__(self, num_neighbours, num_caps, num_filters, new_dim, mode="mean", activation=None,
				 kernel_initializer='glorot_uniform', kernel_regularizer=1e-3,
				 **kwargs):
		super(AggregateLayer, self).__init__(**kwargs)

		self.num_neighbours = num_neighbours
		self.num_caps = num_caps
		self.num_filters = num_filters
		self.new_dim = new_dim
		self.mode = mode
		self.activation = activation
		self.kernel_initializer = kernel_initializer
		self.kernel_regularizer = kernel_regularizer

	def build(self, input_shape):
		'''
		input_shape = [batch_size, Nn, num_input_caps, cap_dim]
		'''

		# self.n_dimension = input_shape[1] 
		# self.num_input_caps = input_shape[2]
		# self.input_dim = input_shape[3]
		self.input_dim = input_shape[2] * input_shape[3]

		initializer = initializers.get(self.kernel_initializer)
		# regularizer = regularizers.get(self.kernel_regularizer)
		regularizer = l2(self.kernel_regularizer)

		self.W = self.add_weight(shape=(self.input_dim, self.num_caps * self.num_filters * self.new_dim), 
									trainable=True, initializer=initializer, regularizer=regularizer,
									name="W")

		self.bias = self.add_weight(shape=(1, self.num_caps * self.num_filters * self.new_dim),
									trainable=True, initializer=initializer, regularizer=regularizer,
									name="bias")

		self.built = True

	def get_config(self):
		config = super(AggregateLayer, self).get_config()
		config.update({"num_neighbours":self.num_neighbours, "num_caps": self.num_caps,
			"num_filters":self.num_filters, "new_dim":self.new_dim, "mode":self.mode, "activation":self.activation,
			"kernel_initializer":self.kernel_initializer, "kernel_regularizer":self.kernel_regularizer})
		return config

	def call(self, inputs):

		'''
		input shape = [None, Nn, num_inputs_caps, cap_dim]
		'''
		
		#aggregate over neighbours


		inputs_shaped = K.reshape(inputs, shape=[-1, self.num_neighbours, self.input_dim])
		# shape is now [batch*Nn+1, num_neighbours, input_dim]
		inputs_aggregated = K.mean(inputs_shaped, axis=1)
		# shape is now [batch*Nn+1, input_dim]
		output = K.dot(inputs_aggregated, self.W) + self.bias
		# shape is not [batch*Nn+1, num_caps*num_filters*new_dim]


		
		if self.activation is not None:
			output = activations.get(self.activation)(output)
		# output = squash(output)

		output = K.reshape(output, shape=[K.shape(inputs)[0], -1, self.num_caps*self.num_filters, self.new_dim])
		
		return output
		
	def compute_output_shape(self, input_shape):
		'''
		input_shape = [None, Nn, num_caps, cap_dim]
		output_shape is [None, Nn+1, num_caps, cap_dim]

		'''
		
		return tuple([input_shape[0], int(input_shape[1] // self.num_neighbours), 
			self.num_caps*self.num_filters, self.new_dim])

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
		self.step_size = int(self.N // (1 + self.num_positive_samples + self.num_negative_samples))
		self.built = True

	def get_config(self):
		config = super(HyperbolicDistanceLayer, self).get_config()
		config.update({"num_positive_samples":self.num_positive_samples, 
					   "num_negative_samples":self.num_negative_samples})
		return config
		
	def safe_norm(self, x, sqrt=False):
		x = K.sum(K.square(x), axis=-1, keepdims=False) + K.epsilon()
		if sqrt:
			x = K.sqrt(x)
		return x
		
	def call(self, inputs):
		'''
		input_shape = [None, N, D]
		'''

		inputs = inputs[:,::self.step_size]
		u = inputs[:,:1]
		v = inputs[:,1:]


		d = tf.acosh(1. + 2. * self.safe_norm(u - v) / 
					 ((1. - self.safe_norm(u)) * (1. - self.safe_norm(v))))
		return d

	def compute_output_shape(self, input_shape):
		return tuple([input_shape[0], self.num_positive_samples + self.num_negative_samples])