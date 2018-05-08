'''

Much of this code is adapted from code written by 
Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
'''

import keras.backend as K
import tensorflow as tf
from keras import initializers, layers, activations#, regularizers
from keras.regularizers import l2
from keras.initializers import RandomUniform

reg = 1e-5

class Length(layers.Layer):
	"""
	Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
	Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
	inputs: shape=[None, num_vectors, dim_vector]
	output: shape=[None, num_vectors]
	
	Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
	"""
	def call(self, inputs):
		return K.sqrt(K.sum(K.square(inputs), axis=-1)) + K.epsilon()

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
	s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True) + K.epsilon()
	scale = s_squared_norm / ((1 + s_squared_norm) * (K.sqrt(s_squared_norm  )))
	return scale * vectors

def embedding_function(vectors, axis=-1):
	"""
	
	"""

	def length(vectors, axis=-1):
	    return K.sqrt(K.sum(K.square(vectors,), axis=axis, keepdims=True, )) + K.epsilon()


	vectors = K.clip(vectors, min_value=K.epsilon(), max_value=1-K.epsilon())

	output = -K.log(vectors)

	# stretch?
	# r = length(output)
	# n = output.shape[-1]

	# phis = []
	# for i in range(n-2):
	#     phi = tf.acos( output[:,:,i,None] / length(output[:,:,i:], axis=-1) ) * 2
	#     # assert (phi.eval() < np.pi).all()
	#     phis.append(phi)
	# phi = tf.acos(  output[:,:,n-2, None] / length(output[:,:,n-2:], axis=-1) ) * tf.sign(output[:,:,-1, None]) * 4
	# # assert (phi.eval() < 2*np.pi).all()
	# phis.append(phi)

	# _output = [r] * n
	# for i in range(n-2):
	#     _output[i] = _output[i] * K.cos(phis[i])
	#     for j in range(i+1,n):
	#         _output[j] = _output[j] *  K.sin(phis[i])
	# _output[n-2] = _output[n-2] * K.cos(phis[-1])
	# _output[n-1] = _output[n-1] * K.sin(phis[-1])

	# output = K.concatenate(_output)

	return squash(output)

class AggGraphCapsuleLayer(layers.Layer):
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
	def __init__(self, num_neighbours, num_capsule, dim_capsule, num_routing=3,
				 kernel_initializer='glorot_uniform', 
				 kernel_regularizer=reg, use_bias=False,
				 **kwargs):
		super(AggGraphCapsuleLayer, self).__init__(**kwargs)
		self.num_neighbours = num_neighbours
		self.num_capsule = num_capsule
		self.dim_capsule = dim_capsule
		self.num_routing = num_routing
		self.kernel_initializer = kernel_initializer#RandomUniform(minval=-1e-8, maxval=1e-8)#kernel_initializer
		self.kernel_regularizer = kernel_regularizer
		self.use_bias = use_bias

	def build(self, input_shape):
		assert len(input_shape) == 4, "The input Tensor should have shape=[None, N, input_num_capsule, input_dim_capsule]"

		self.input_num_capsule = input_shape[2]
		self.input_dim_capsule = input_shape[3]
		
		initializer = initializers.get(self.kernel_initializer)
		# regularizer = regularizers.get(self.kernel_regularizer)
		regularizer = l2(self.kernel_regularizer)

		# Transform matrix
		self.W = self.add_weight(shape=[self.input_num_capsule, self.input_dim_capsule,
										self.num_capsule * self.dim_capsule],
								 initializer=initializer,
								 regularizer=regularizer,
								 name='W')
		if self.use_bias:
			self.bias = self.add_weight(shape=[self.input_num_capsule, self.num_capsule * self.dim_capsule],
									initializer=initializer,
									regularizer=regularizer,
									name="bias")

		self.built = True

	def get_config(self):
		config = super(AggGraphCapsuleLayer, self).get_config()
		config.update({"num_neighbours":self.num_neighbours, "num_capsule":self.num_capsule,
			"dim_capsule":self.dim_capsule, "num_routing":self.num_routing, "use_bias": self.use_bias,
			"kernel_initializer":self.kernel_initializer, "kernel_regularizer":self.kernel_regularizer})
		return config

	def call(self, inputs):
		# inputs.shape=[None, N, input_num_capsule, input_dim_capsule]

		# inputs shape is [None, N, input_num_caps, input_cap_dim]
		batch_size = K.shape(inputs)[0]
		# N = K.shape(inputs)[1]

		inputs = K.reshape(inputs, [-1, self.input_num_capsule, self.input_dim_capsule])
		# shape is now [None*N, input num caps, input cap dim]
		inputs = K.permute_dimensions(inputs, [1, 0, 2])
		# shape is now [num_neighbours * input num caps, None*N, input cap dim]
		# W shape is [input num caps, input cap dim, num_caps * cap dim]
		inputs_hat = K.batch_dot(inputs, self.W, axes=[2, 1])
		# shape is now [input num caps, None*N, num caps * cap dim]
		inputs_hat = K.permute_dimensions(inputs_hat, [1, 0, 2])
		# shape is now [None*N, input num caps, num_caps * caps dim]

		if self.use_bias:
			inputs_hat = K.bias_add(inputs_hat, self.bias)

		inputs_hat = K.reshape(inputs_hat, tf.stack([-1, self.num_neighbours * self.input_num_capsule, 
													self.num_capsule, self.dim_capsule]))

		# shape is now [None*Nn+1, num_neighbours * input num caps, num caps, cap dim]

		inputs_hat = K.permute_dimensions(inputs_hat, pattern=[0,2,1,3])
		# shape is now [None* Nn+1, num_caps, num_neighbours*num_input_caps, cap_dim]

		
		# Begin: Routing algorithm ---------------------------------------------------------------------#
		# In forward pass, `inputs_hat_stopped` = `inputs_hat`;
		# In backward, no gradient can flow from `inputs_hat_stopped` back to `inputs_hat`.
		inputs_hat_stopped = K.stop_gradient(inputs_hat)
		
		# The prior for coupling coefficient, initialized as zeros.
		# b.shape = [None* Nn+1, self.num_capsule, self.input_num_capsule].
		# b = tf.zeros(shape=[batch_size * int(N // self.num_neighbours),
							# self.num_capsule, self.num_neighbours * self.input_num_capsule])
		b = K.zeros_like(inputs_hat[:,:,:,0])
		# print b.shape
		# raise SystemExit

		# assert self.num_routing > 0, 'The num_routing should be > 0.'
		for i in range(self.num_routing):
			# c.shape=[batch_size*N, num_capsule, num_neighbours*input_num_capsule]
			c = tf.nn.softmax(b, dim=1)

			# At last iteration, use `inputs_hat` to compute `outputs` in order to backpropagate gradient
			if i == self.num_routing - 1:
				# c.shape =  [batch_size* N, num_capsule, input_num_capsule]
				# inputs_hat.shape=[None* N, num_capsule, input_num_capsule, dim_capsule]
				# The first three dimensions as `batch` dimension,
				# then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
				# outputs.shape=[None* N, num_capsule, dim_capsule]

				# outputs = squash(K.batch_dot(c, inputs_hat, axes=[2, 2]))  
				outputs = K.batch_dot(c, inputs_hat, axes=[2, 2])
			else:  # Otherwise, use `inputs_hat_stopped` to update `b`. No gradients flow on this path.
				outputs = squash(K.batch_dot(c, inputs_hat_stopped, axes=[2, 2]))
				# outputs.shape =  [None* N, num_capsule, dim_capsule]
				# inputs_hat.shape=[None* N, num_capsule, input_num_capsule, dim_capsule]
				# The first two dimensions as `batch` dimension,
				# then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
				# b.shape=[batch_size* N, num_capsule, num_neighbours*input_num_capsule]
				b += K.batch_dot(outputs, inputs_hat_stopped, axes=[2,3]) 
		# End: Routing algorithm -----------------------------------------------------------------------#

		outputs = K.reshape(outputs, 
			shape=tf.stack([batch_size, -1, self.num_capsule, self.dim_capsule]) )

		return outputs

	def compute_output_shape(self, input_shape):
		return tuple([input_shape[0], int(input_shape[1] // self.num_neighbours)] + [self.num_capsule, self.dim_capsule])

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
				 kernel_regularizer=reg, use_bias=False,
				 **kwargs):
		super(GraphCapsuleLayer, self).__init__(**kwargs)
		self.num_capsule = num_capsule
		self.dim_capsule = dim_capsule
		self.num_routing = num_routing
		self.kernel_initializer = kernel_initializer#RandomUniform(minval=-0.5, maxval=0.5)#kernel_initializer
		self.kernel_regularizer = kernel_regularizer
		self.use_bias = use_bias

	def build(self, input_shape):
		assert len(input_shape) == 4, "The input Tensor should have shape=[None, N, input_num_capsule, input_dim_capsule]"
		# self.batch_size = input_shape[0]
		# self.neighbours = input_shape[1]
		self.input_num_capsule = input_shape[2]
		self.input_dim_capsule = input_shape[3]
		
		initializer = initializers.get(self.kernel_initializer)
		# regularizer = regularizers.get(self.kernel_regularizer)
		regularizer = l2(self.kernel_regularizer)

		# Transform matrix
		# self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
		#                               self.dim_capsule, self.input_dim_capsule],
		#                        initializer=initializer,
		#                        regularizer=regularizer,
		#                        name='W')
		self.W = self.add_weight(shape=[self.input_num_capsule, self.input_dim_capsule,
										self.num_capsule * self.dim_capsule],
								 initializer=initializer,
								 regularizer=regularizer,
								 name='W')
		if self.use_bias:
			self.bias = self.add_weight(shape=[self.input_num_capsule, self.num_capsule * self.dim_capsule],
									initializer=initializer,
									regularizer=regularizer,
									name="bias")

		self.built = True

	def get_config(self):
		config = super(GraphCapsuleLayer, self).get_config()
		config.update({"num_capsule":self.num_capsule,
			"dim_capsule":self.dim_capsule, "num_routing":self.num_routing, "use_bias": self.use_bias,
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
		#                     K.map_fn(lambda y: K.batch_dot(y, self.W, [2, 3]), elems=x), elems=inputs_tiled)

		# inputs_hat = K.map_fn(lambda x: 
		#   # K.map_fn(lambda y: 
		#   K.reshape(K.batch_dot(x, self.W, axes=[2, 1]), 
		#       shape=[-1, self.input_num_capsule, self.num_capsule, self.dim_capsule]), 
		#   # elems=x),
		#   elems=inputs)
		# inputs_hat = K.map_fn(lambda x, shape=[-1, self.num_capsule, self.dim_capsule]: 
		#   K.map_fn(lambda y, shape=shape: 
		#   K.reshape(y, shape=shape), elems=x), 
		#   elems=inputs_hat)


		# inputs shape is [None, N, input_num_caps, input_cap_dim]
		batch_size = K.shape(inputs)[0]
		num_neighbours = K.shape(inputs)[1]


		inputs = K.reshape(inputs, [-1, self.input_num_capsule, self.input_dim_capsule])
		# shape is now [None*N, input num caps, input cap dim]
		inputs = K.permute_dimensions(inputs, [1, 0, 2])
		# shape is now [input num caps, None*N, input cap dim]
		# W shape is [input num caps, input cap dim, num_caps * cap dim]
		inputs_hat = K.batch_dot(inputs, self.W, axes=[2, 1])
		# shape is now [input num caps, None*N, num caps * cap dim]
		inputs_hat = K.permute_dimensions(inputs_hat, [1, 0, 2])
		# shape is now [None*N, input num caps, num_caps * caps dim]

		if self.use_bias:
			inputs_hat = K.bias_add(inputs_hat, self.bias)

		inputs_hat = K.reshape(inputs_hat, tf.stack([-1, self.input_num_capsule, 
													self.num_capsule, self.dim_capsule]))

		# shape is now [None*N, input num caps, num caps, cap dim]

		inputs_hat = K.permute_dimensions(inputs_hat, pattern=[0,2,1,3])
		# shape is now [None* N, num_caps, num_input_caps, cap_dim]
		# print "inputs_hat", inputs_hat.shape
		# raise SystemExit
		
		# Begin: Routing algorithm ---------------------------------------------------------------------#
		# In forward pass, `inputs_hat_stopped` = `inputs_hat`;
		# In backward, no gradient can flow from `inputs_hat_stopped` back to `inputs_hat`.
		inputs_hat_stopped = K.stop_gradient(inputs_hat)
		
		# The prior for coupling coefficient, initialized as zeros.
		# b.shape = [None, N, self.num_capsule, self.input_num_capsule].
		b = tf.zeros(shape=[batch_size * num_neighbours,
							self.num_capsule, self.input_num_capsule])

		# assert self.num_routing > 0, 'The num_routing should be > 0.'
		for i in range(self.num_routing):
			# c.shape=[batch_size*N, num_capsule, input_num_capsule]
			c = tf.nn.softmax(b, dim=1)

			# At last iteration, use `inputs_hat` to compute `outputs` in order to backpropagate gradient
			if i == self.num_routing - 1:
				# c.shape =  [batch_size* N, num_capsule, input_num_capsule]
				# inputs_hat.shape=[None* N, num_capsule, input_num_capsule, dim_capsule]
				# The first three dimensions as `batch` dimension,
				# then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
				# outputs.shape=[None* N, num_capsule, dim_capsule]

				outputs = squash(K.batch_dot(c, inputs_hat, axes=[2, 2]))  
				# outputs = K.batch_dot(c, inputs_hat, axes=3)
			else:  # Otherwise, use `inputs_hat_stopped` to update `b`. No gradients flow on this path.
				outputs = squash(K.batch_dot(c, inputs_hat_stopped, axes=[2, 2]))
				# outputs.shape =  [None* N, num_capsule, dim_capsule]
				# inputs_hat.shape=[None* N, num_capsule, input_num_capsule, dim_capsule]
				# The first two dimensions as `batch` dimension,
				# then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
				# b.shape=[batch_size* N, num_capsule, input_num_capsule]
				b += K.batch_dot(outputs, inputs_hat_stopped, axes=[2,3])
		# End: Routing algorithm -----------------------------------------------------------------------#

		outputs = K.reshape(outputs, shape=tf.stack([batch_size, num_neighbours, 
			self.num_capsule, self.dim_capsule]) )

		return outputs

	def compute_output_shape(self, input_shape):
		# print "graphcaps shape", tuple(list(input_shape[:2]) + [self.num_capsule, self.dim_capsule])
		return tuple(list(input_shape[:2]) + [self.num_capsule, self.dim_capsule])

class SimpleAggregateLayer(layers.Layer):

	def __init__(self, num_neighbours, mode="length", **kwargs):
		super(SimpleAggregateLayer, self).__init__(**kwargs)

		self.num_neighbours = num_neighbours
		self.mode = mode

	def build(self, input_shape):
		'''
		input_shape = [batch_size, Nn, input_dim]
		'''

		assert len(input_shape) == 4, "incorrect simple aggregation input"

		# print "input shape", input_shape
		# self.n_dimension = int(input_shape[1] // self.num_neighbours)
		# print "n dimension",self.n_dimension

		self.num_input_caps = input_shape[2]
		self.input_cap_dim = input_shape[3]
		# self.input_dim = input_shape[2]

		# initializer = initializers.get(self.kernel_initializer)
		# # regularizer = regularizers.get(self.kernel_regularizer)
		# regularizer = l2(self.kernel_regularizer)

		# self.W = self.add_weight(shape=(self.input_dim, self.num_caps * self.num_filters * self.new_dim), 
		#                           trainable=True, initializer=initializer, regularizer=regularizer,
		#                           name="W")

		# self.bias = self.add_weight(shape=(self.num_caps * self.num_filters * self.new_dim, ),
		#                           trainable=True, initializer=initializer, regularizer=regularizer,
		#                           name="bias")

		self.built = True

	def get_config(self):
		config = super(SimpleAggregateLayer, self).get_config()
		config.update({"num_neighbours":self.num_neighbours, "mode":self.mode,})
		return config

	def length(self, inputs, axis=-1):
		return K.sqrt(K.sum(K.square(inputs), axis=axis, keepdims=False) + K.epsilon())# + K.epsilon()

	def call(self, inputs):

		'''
		input shape = [None, Nn, num_inputs_caps, cap_dim]
		'''
		
		#aggregate over neighbours
		if self.mode == "mean":
			agg_fn = K.mean
		elif self.mode == "max":
			agg_fn = K.max
		elif self.mode == "length":
			agg_fn = self.length
		else:
			raise Exception

		inputs_shaped = K.reshape(inputs, shape=[-1, self.num_neighbours, 
			self.num_input_caps, self.input_cap_dim])

		output = agg_fn(inputs_shaped, axis=1)

		output = K.reshape(output, shape=tf.stack([K.shape(inputs)[0], -1, self.num_input_caps, self.input_cap_dim]))

		return output
		
	def compute_output_shape(self, input_shape):
		'''
		input_shape = [None, Nn, num_caps, cap_dim]
		output_shape is [None, Nn+1, num_caps, cap_dim]

		'''     
		return tuple([input_shape[0], int(input_shape[1] // self.num_neighbours), 
			self.num_input_caps, self.input_cap_dim])

class AggregateLayer(layers.Layer):
	"""
	A layer to perform a convolutional filter on a graph with mean aggregation
	TODO
	
	Author: David McDonald, Email: `dxm237@cs.bham.ac.uk'
	"""
	def __init__(self, num_neighbours, num_caps, num_filters, new_dim, mode="mean", activation=None,
				 kernel_initializer='glorot_uniform', kernel_regularizer=reg,
				 **kwargs):
		super(AggregateLayer, self).__init__(**kwargs)

		self.num_neighbours = num_neighbours
		self.num_caps = num_caps
		self.num_filters = num_filters
		self.new_dim = new_dim
		self.mode = mode
		self.activation = activation
		self.kernel_initializer = kernel_initializer#RandomUniform(minval=-0.5, maxval=0.5)#kernel_initializer
		self.kernel_regularizer = kernel_regularizer

	def build(self, input_shape):
		'''
		input_shape = [batch_size, Nn, input_dim]
		'''
		assert len(input_shape) == 3, "incorrect aggregation input"

		# self.n_dimension = input_shape[1] 
		# self.num_input_caps = input_shape[2]
		# self.input_dim = input_shape[3]
		self.input_dim = input_shape[2]

		initializer = initializers.get(self.kernel_initializer)
		# regularizer = regularizers.get(self.kernel_regularizer)
		regularizer = l2(self.kernel_regularizer)

		self.W = self.add_weight(shape=(self.input_dim, self.num_caps * self.num_filters * self.new_dim), 
									trainable=True, initializer=initializer, regularizer=regularizer,
									name="W")

		self.bias = self.add_weight(shape=(self.num_caps * self.num_filters * self.new_dim, ),
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
		# print inputs_shaped.shape
		# raise SystemExit
		# shape is now [batch*Nn+1, num_neighbours, input_dim]
		if self.mode == "mean":
			agg_fn = K.mean
		else:
			raise Exception
		inputs_aggregated = agg_fn(inputs_shaped, axis=1)
		# print K.eval(inputs_aggregated)
		# raise SystemExit
		# shape is now [batch*Nn+1, input_dim]
		output = K.dot(inputs_aggregated, self.W)# + self.bias
		# print output.shape
		output = K.bias_add(output, self.bias)
		# print output.shape
		# raise SystemExit
		# shape is not [batch*Nn+1, num_caps*num_filters*new_dim]


		
		# if self.activation is not None:
		#   output = activations.get(self.activation)(output)
		# output = squash(output)

		output = K.reshape(output, shape=tf.stack([K.shape(inputs)[0], -1, self.num_caps*self.num_filters, self.new_dim]))
		output = squash(output)
		# print output.shape
		# raise SystemExit
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
		assert input_shape is not None
		self.N = input_shape[1]
		self.step_size = int(self.N // (1 + self.num_positive_samples + self.num_negative_samples))
		self.built = True
		# print input_shape, self.N, self.num_positive_samples, self.num_negative_samples, self.step_size
		# print range(0, self.N, self.step_size)

		# print input_shape, self.N, self.step_size, range(0, self.N, self.step_size)

	def get_config(self):
		config = super(HyperbolicDistanceLayer, self).get_config()
		config.update({"num_positive_samples":self.num_positive_samples, 
					   "num_negative_samples":self.num_negative_samples})
		return config
		
	def safe_norm(self, x, axis=-1, sqrt=False, clip=False):
		y = K.sum(K.square(x), axis=axis, keepdims=False) + K.epsilon()
		if sqrt:
			y = K.sqrt(y)
		if clip:
			y = K.clip(y, min_value=K.epsilon(), max_value=1-K.epsilon())
		return y
		
	def call(self, inputs):
		'''
		input_shape = [None, N, D]
		'''

		inputs = inputs[:,0:self.N:self.step_size]
		# print inputs.shape, K.shape(inputs)[1], self.step_size
		# raise SystemExit
		u = inputs[:,:1]
		v = inputs[:,1:]


		d = tf.acosh(1. + 2. * self.safe_norm(u - v) / 
					 ((1. - self.safe_norm(u, clip=True)) * (1. - self.safe_norm(v, clip=True))))
		# d = self.safe_norm(u - v, sqrt=True)
		return d

	def compute_output_shape(self, input_shape):
		# print "hyp dist", tuple([input_shape[0], self.num_positive_samples + self.num_negative_samples])
		return tuple([input_shape[0], self.num_positive_samples + self.num_negative_samples])