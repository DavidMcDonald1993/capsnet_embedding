'''

Much of this code is adapted from code written by 
Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
'''

import numpy as np

import keras.backend as K
import tensorflow as tf
from keras import initializers, layers, activations#, regularizers
from keras.regularizers import l2
from keras.initializers import RandomUniform
reg = 1e-30

from tensorflow.contrib.distributions.python.ops.relaxed_onehot_categorical import RelaxedOneHotCategorical

# min_norm = np.nextafter(0, 1, dtype=K.floatx())
max_norm = np.nextafter(1, 0, dtype=K.floatx())
max_ = np.finfo(K.floatx()).max
min_norm = 1e-7
# max_norm = 0.999
# max_ = 1e+4
# raise SystemExit

class Length(layers.Layer):
	"""
	Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
	Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
	inputs: shape=[None, num_vectors, dim_vector]
	output: shape=[None, num_vectors]
	
	Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
	"""
	def __init__(self, **kwargs):
		super(Length, self).__init__(**kwargs)

	def call(self, inputs):
		outputs = K.sqrt(K.sum(K.square(inputs), axis=-1))
		# return K.clip(outputs, min_value=K.epsilon(), max_value=1-K.epsilon())
		return outputs

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
	s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)# + 1e-32
	scale = s_squared_norm / (1 + s_squared_norm) #(K.sqrt(s_squared_norm  )))
	scale = tf.clip_by_value(scale, min_norm, max_norm)
	s_norm = K.clip(K.sqrt(s_squared_norm), min_value=min_norm, max_value=max_  )
	outputs = scale * vectors / s_norm
	# outputs = K.clip(outputs, min_value=min_norm, max_value=max_norm)
	return outputs

def embedding_function(vectors, axis=-1):
	"""
	
	"""

	output = vectors
	# output = -K.log(output)

	return squash(output, axis=axis)



class UHatLayer(layers.Layer):

	def __init__(self, num_capsule, dim_capsule, kernel_initializer='glorot_uniform', 
				 kernel_regularizer=reg, use_bias=False, **kwargs):
		super(UHatLayer, self).__init__(**kwargs)
		self.num_capsule = num_capsule
		self.dim_capsule = dim_capsule
		self.kernel_initializer = kernel_initializer
		self.kernel_regularizer = kernel_regularizer
		self.use_bias = use_bias

	def build(self, input_shape):
		assert len(input_shape) == 3, "The input Tensor should have shape=[None, input num caps, input cap dim]"

		self.input_num_capsule = input_shape[1]
		self.input_dim_capsule = input_shape[2]
		
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
		config = super(UHatLayer, self).get_config()
		config.update({"num_capsule":self.num_capsule,
			"dim_capsule":self.dim_capsule, "use_bias": self.use_bias,
			"kernel_initializer":self.kernel_initializer, "kernel_regularizer":self.kernel_regularizer})
		return config

	def call(self, inputs):
		# inputs.shape=[None,input_num_capsule, input_dim_capsule]

		# inputs shape is [None, input_num_caps, input_cap_dim]
		batch_size = K.shape(inputs)[0]
		# shape is now [None, input num caps, input cap dim]
		inputs = K.permute_dimensions(inputs, [1, 0, 2])
		# shape is now [input num caps, None, input cap dim]

		# W shape is [input num caps, input cap dim, num_caps * cap dim]
		inputs_hat = K.batch_dot(inputs, self.W, axes=[2, 1])
		# shape is now [input num caps, None, num caps * cap dim]
		inputs_hat = K.permute_dimensions(inputs_hat, [1, 0, 2])
		# shape is now [None, input num caps, num_caps * caps dim]

		if self.use_bias:
			inputs_hat = K.bias_add(inputs_hat, self.bias)

		inputs_hat = K.reshape(inputs_hat, tf.stack([-1, self.input_num_capsule, 
													self.num_capsule, self.dim_capsule]))
		# shape is now [None, input num caps, num caps, cap dim]
		return inputs_hat

		# inputs_hat = K.permute_dimensions(inputs_hat, pattern=[0,2,1,3])
		# # shape is now [None, num_caps, num_input_caps, cap_dim]
	
	def compute_output_shape(self, input_shape):
		return input_shape[0], self.input_num_capsule, self.num_capsule, self.dim_capsule


class NeighbourhoodSamplingLayer(layers.Layer):


	def __init__(self,   sample_size, **kwargs):

		super(NeighbourhoodSamplingLayer, self).__init__(**kwargs)
		self.sample_size = sample_size

		# assert type(neighbours) == list

		# if not sample_size is None:
		# 	self.neighbours = self.pad_neighbours(neighbours, sample_size)
		# else:
		# 	raise Exception
		# 	# not implemented
		# 	# self.neighbours = neighbours

		# pass

	def build(self, input_shape):
		# print input_shape
		x_shape, adj_input_shape = input_shape
		assert len(x_shape) == 4, "The input Tensor should have shape=[None, num_input_caps, num_caps, cap_dim]"
		assert len(adj_input_shape) == 2, "adj_input should have shape [None, max_num_neighbours]"
		self.num_input_caps = x_shape[1]
		self.num_caps = x_shape[2]
		self.cap_dim = x_shape[3]
		self.built = True

	def get_config(self):
		config = super(NeighbourhoodSamplingLayer, self).get_config()
		config.update({ "sample_size": self.sample_size})
		return config

	def call(self, inputs):

		inputs, adj_input = inputs

		# nodes = K.arange(K.shape(inputs)[0])
		# nodes = K.reshape(nodes, [-1, 1])
		nodes = adj_input[:,:1]
		adj_input = adj_input[:,1:]
		# print nodes.shape
		# neighbour_samples = K.concatenate([[tf.random_shuffle(n)[:self.sample_size]] for n in self.neighbours], axis=0)
		# print neighbour_samples.shape
		neighbour_samples = K.transpose(adj_input)
		neighbour_samples = tf.random_shuffle(neighbour_samples)
		neighbour_samples = neighbour_samples[:self.sample_size]
		neighbour_samples = K.transpose(neighbour_samples)
		ids = K.concatenate([nodes, neighbour_samples])
		ids = K.cast(ids, tf.int32)
		# ids = K.map_fn(lambda x : tf.random_shuffle(x)[:self.sample_size], 
		# 	elems=self.neighbours)
		X = tf.nn.embedding_lookup(params=inputs, ids=ids)

		# print X.shape
		# raise SystemExit
		# print K.reshape(X, [-1, (self.sample_size+1)*self.num_input_caps,
		# 	self.num_caps, self.cap_dim]).shape
		# raise SystemExit

		return K.reshape(X, [-1, (self.sample_size+1)*self.num_input_caps,
			self.num_caps, self.cap_dim])

	def compute_output_shape(self, input_shape):
		input_shape = input_shape[0]
		return input_shape[0], (self.sample_size+1) * input_shape[1], input_shape[2], input_shape[3]

class DynamicRoutingLayer(layers.Layer):


	def __init__(self, num_routing, **kwargs):
		super(DynamicRoutingLayer, self).__init__(**kwargs)
		self.num_routing = num_routing

	def build(self, input_shape):
		assert len(input_shape) == 4, "The input Tensor should have shape=[None, num_samples*num_input_caps, num_caps, cap_dim"

		self.built = True

	def get_config(self):
		config = super(DynamicRoutingLayer, self).get_config()
		config.update({"num_routing": self.num_routing})
		return config

	def call(self, inputs):

		# inputs shape = [None, num_samples*num_input_caps, num_caps, cap_dim]
		
		inputs_hat = K.permute_dimensions(inputs, pattern=[0,2,1,3])
		# shape is now [None, num_caps, num_samples*num_input_caps, cap_dim]
		# print "ion hat", inputs_hat.shape

		inputs_hat_stopped = K.stop_gradient(inputs_hat)
		
		# The prior for coupling coefficient, initialized as zeros.
		# b.shape = [None, self.num_capsule, self.input_num_capsule].
		b = K.zeros_like(inputs_hat[:,:,:,0])
		# print "b", b.shape

		# assert self.num_routing > 0, 'The num_routing should be > 0.'
		for i in range(self.num_routing):
			# c.shape=[batch_size*Nn+1, num_capsule, num_neighbours*input_num_capsule]
			c = tf.nn.softmax(b, dim=1)
			# print "c", c.shape

			# At last iteration, use `inputs_hat` to compute `outputs` in order to backpropagate gradient
			if i == self.num_routing - 1:
				# c.shape =  [batch_size, num_capsule, input_num_capsule]
				# inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
				# The first three dimensions as `batch` dimension,
				# then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
				# outputs.shape=[None, num_capsule, dim_capsule]

				outputs = squash(K.batch_dot(c, inputs_hat, axes=[2, 2]))  
			else:  # Otherwise, use `inputs_hat_stopped` to update `b`. No gradients flow on this path.
				outputs = squash(K.batch_dot(c, inputs_hat_stopped, axes=[2, 2]))
				# outputs.shape =  [None, num_capsule, dim_capsule]
				# inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
				# The first two dimensions as `batch` dimension,
				# then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
				# b.shape=[batch_size, num_capsule, num_neighbours*input_num_capsule]
				b += K.batch_dot(outputs, inputs_hat_stopped, axes=[2,3]) 
		# End: Routing algorithm -----------------------------------------------------------------------#

		return outputs

	def compute_output_shape(self, input_shape):
		# output shape is [None, num_caps, cap_dim]
		return input_shape[0], input_shape[2], input_shape[3]

class Mask(layers.Layer):
	"""
	Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional 
	input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
	masked Tensor.
	For example:
		```
		x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
		y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
		out = Mask()(x)  # out.shape=[8, 6]
		# or
		out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
		```
	"""
	def call(self, inputs, **kwargs):
		if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
			assert len(inputs) == 2
			inputs, mask = inputs
		else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
			# compute lengths of capsules
			x = K.sqrt(K.sum(K.square(inputs), axis=-1, keepdims=False))
			# print x.shape
			# raise SystemExit
			# generate the mask which is a one-hot code.
			# mask.shape=[None, n_classes]=[None, num_capsule]
			# mask = K.one_hot(indices=K.argmax(x, axis=1), num_classes=x.get_shape().as_list()[1])
			# mask = K.expand_dims(mask, axis=-1)
			x = K.log(x / (1-x))
			# mask = tf.nn.softmax(x, axis=1)
			dist = RelaxedOneHotCategorical(logits=x, temperature=0.001)
			mask = dist.sample()
			mask = K.expand_dims(mask, axis=-1)

		# inputs.shape=[None, num_capsule, dim_capsule]
		# mask.shape=[None, num_capsule]
		# masked.shape=[None, num_capsule * dim_capsule]
		# print K.expand_dims(mask, -1).shape
		# raise SystemExit
		masked = K.batch_flatten(mask * inputs)
		return masked

	def compute_output_shape(self, input_shape):
		if type(input_shape[0]) is tuple:  # true label provided
			return tuple([None, input_shape[0][1] * input_shape[0][2]])
		else:  # no true label provided
			return tuple([None, input_shape[1] * input_shape[2]])

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
		# shape is now [input num caps, None*N, input cap dim]
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
			# c.shape=[batch_size*Nn+1, num_capsule, num_neighbours*input_num_capsule]
			c = tf.nn.softmax(b, dim=1)

			# At last iteration, use `inputs_hat` to compute `outputs` in order to backpropagate gradient
			if i == self.num_routing - 1:
				# c.shape =  [batch_size* N, num_capsule, input_num_capsule]
				# inputs_hat.shape=[None* N, num_capsule, input_num_capsule, dim_capsule]
				# The first three dimensions as `batch` dimension,
				# then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
				# outputs.shape=[None* N, num_capsule, dim_capsule]

				outputs = squash(K.batch_dot(c, inputs_hat, axes=[2, 2]))  
				# outputs = K.batch_dot(c, inputs_hat, axes=[2, 2])
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
		# return lambda input_shape : tuple([input_shape[0], int(input_shape[1] / self.num_neighbours)] +\
		#  [self.num_capsule, self.dim_capsule])
		return tuple([input_shape[0], int(input_shape[1] / self.num_neighbours)] + [self.num_capsule, self.dim_capsule])

# class GraphCapsuleLayer(layers.Layer):
# 	"""
# 	The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the 
# 	neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
# 	from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
# 	[None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.
	
# 	:param num_capsule: number of capsules in this layer
# 	:param dim_capsule: dimension of the output vectors of the capsules in this layer
# 	:param num_routing: number of iterations for the routing algorithm

# 	Author: David McDonald, E-mail: `dxm237@cs.bham.ac.uk`, Github: `https://github.com/DavidMcDonald1993/capsnet_embedding.git`
# 	"""
# 	def __init__(self, num_capsule, dim_capsule, num_routing=3,
# 				 kernel_initializer='glorot_uniform', 
# 				 kernel_regularizer=reg, use_bias=False,
# 				 **kwargs):
# 		super(GraphCapsuleLayer, self).__init__(**kwargs)
# 		self.num_capsule = num_capsule
# 		self.dim_capsule = dim_capsule
# 		self.num_routing = num_routing
# 		self.kernel_initializer = kernel_initializer#RandomUniform(minval=-0.5, maxval=0.5)#kernel_initializer
# 		self.kernel_regularizer = kernel_regularizer
# 		self.use_bias = use_bias

# 	def build(self, input_shape):
# 		assert len(input_shape) == 4, "The input Tensor should have shape=[None, N, input_num_capsule, input_dim_capsule]"
# 		# self.batch_size = input_shape[0]
# 		# self.neighbours = input_shape[1]
# 		self.input_num_capsule = input_shape[2]
# 		self.input_dim_capsule = input_shape[3]
		
# 		initializer = initializers.get(self.kernel_initializer)
# 		# regularizer = regularizers.get(self.kernel_regularizer)
# 		regularizer = l2(self.kernel_regularizer)

# 		# Transform matrix
# 		# self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
# 		#                               self.dim_capsule, self.input_dim_capsule],
# 		#                        initializer=initializer,
# 		#                        regularizer=regularizer,
# 		#                        name='W')
# 		self.W = self.add_weight(shape=[self.input_num_capsule, self.input_dim_capsule,
# 										self.num_capsule * self.dim_capsule],
# 								 initializer=initializer,
# 								 regularizer=regularizer,
# 								 name='W')
# 		if self.use_bias:
# 			self.bias = self.add_weight(shape=[self.input_num_capsule, self.num_capsule * self.dim_capsule],
# 									initializer=initializer,
# 									regularizer=regularizer,
# 									name="bias")

# 		self.built = True

# 	def get_config(self):
# 		config = super(GraphCapsuleLayer, self).get_config()
# 		config.update({"num_capsule":self.num_capsule,
# 			"dim_capsule":self.dim_capsule, "num_routing":self.num_routing, "use_bias": self.use_bias,
# 			"kernel_initializer":self.kernel_initializer, "kernel_regularizer":self.kernel_regularizer})
# 		return config

# 	def call(self, inputs):
# 		# inputs.shape=[None, N, input_num_capsule, input_dim_capsule]
# 		# inputs_expand.shape=[None, N, 1, input_num_capsule, input_dim_capsule]
# 		# inputs_expand = K.expand_dims(inputs, 2)

# 		# Replicate num_capsule dimension to prepare being multiplied by W
# 		# inputs_tiled.shape=[None, N, num_capsule, input_num_capsule, input_dim_capsule]
# 		# inputs_tiled = K.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])

# 		# Compute `inputs * W` by scanning inputs_tiled on dimension 0.
# 		# y.shape=[num_capsule, input_num_capsule, input_dim_capsule]
# 		# W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
# 		# Regard the first two dimensions as `batch` dimension,
# 		# then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
# 		# inputs_hat.shape = [None, N, num_capsule, input_num_capsule, dim_capsule]
# 		# inputs_hat = K.map_fn(lambda x: 
# 		#                     K.map_fn(lambda y: K.batch_dot(y, self.W, [2, 3]), elems=x), elems=inputs_tiled)

# 		# inputs_hat = K.map_fn(lambda x: 
# 		#   # K.map_fn(lambda y: 
# 		#   K.reshape(K.batch_dot(x, self.W, axes=[2, 1]), 
# 		#       shape=[-1, self.input_num_capsule, self.num_capsule, self.dim_capsule]), 
# 		#   # elems=x),
# 		#   elems=inputs)
# 		# inputs_hat = K.map_fn(lambda x, shape=[-1, self.num_capsule, self.dim_capsule]: 
# 		#   K.map_fn(lambda y, shape=shape: 
# 		#   K.reshape(y, shape=shape), elems=x), 
# 		#   elems=inputs_hat)


# 		# inputs shape is [None, N, input_num_caps, input_cap_dim]
# 		batch_size = K.shape(inputs)[0]
# 		num_neighbours = K.shape(inputs)[1]


# 		inputs = K.reshape(inputs, [-1, self.input_num_capsule, self.input_dim_capsule])
# 		# shape is now [None*N, input num caps, input cap dim]
# 		inputs = K.permute_dimensions(inputs, [1, 0, 2])
# 		# shape is now [input num caps, None*N, input cap dim]
# 		# W shape is [input num caps, input cap dim, num_caps * cap dim]
# 		inputs_hat = K.batch_dot(inputs, self.W, axes=[2, 1])
# 		# shape is now [input num caps, None*N, num caps * cap dim]
# 		inputs_hat = K.permute_dimensions(inputs_hat, [1, 0, 2])
# 		# shape is now [None*N, input num caps, num_caps * caps dim]

# 		if self.use_bias:
# 			inputs_hat = K.bias_add(inputs_hat, self.bias)

# 		inputs_hat = K.reshape(inputs_hat, tf.stack([-1, self.input_num_capsule, 
# 													self.num_capsule, self.dim_capsule]))

# 		# shape is now [None*N, input num caps, num caps, cap dim]

# 		inputs_hat = K.permute_dimensions(inputs_hat, pattern=[0,2,1,3])
# 		# shape is now [None* N, num_caps, num_input_caps, cap_dim]
# 		# print "inputs_hat", inputs_hat.shape
# 		# raise SystemExit
		
# 		# Begin: Routing algorithm ---------------------------------------------------------------------#
# 		# In forward pass, `inputs_hat_stopped` = `inputs_hat`;
# 		# In backward, no gradient can flow from `inputs_hat_stopped` back to `inputs_hat`.
# 		inputs_hat_stopped = K.stop_gradient(inputs_hat)
		
# 		# The prior for coupling coefficient, initialized as zeros.
# 		# b.shape = [None, N, self.num_capsule, self.input_num_capsule].
# 		b = tf.zeros(shape=[batch_size * num_neighbours,
# 							self.num_capsule, self.input_num_capsule])

# 		# assert self.num_routing > 0, 'The num_routing should be > 0.'
# 		for i in range(self.num_routing):
# 			# c.shape=[batch_size*N, num_capsule, input_num_capsule]
# 			c = tf.nn.softmax(b, dim=1)

# 			# At last iteration, use `inputs_hat` to compute `outputs` in order to backpropagate gradient
# 			if i == self.num_routing - 1:
# 				# c.shape =  [batch_size* N, num_capsule, input_num_capsule]
# 				# inputs_hat.shape=[None* N, num_capsule, input_num_capsule, dim_capsule]
# 				# The first three dimensions as `batch` dimension,
# 				# then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
# 				# outputs.shape=[None* N, num_capsule, dim_capsule]

# 				outputs = squash(K.batch_dot(c, inputs_hat, axes=[2, 2]))  
# 				# outputs = K.batch_dot(c, inputs_hat, axes=3)
# 			else:  # Otherwise, use `inputs_hat_stopped` to update `b`. No gradients flow on this path.
# 				outputs = squash(K.batch_dot(c, inputs_hat_stopped, axes=[2, 2]))
# 				# outputs.shape =  [None* N, num_capsule, dim_capsule]
# 				# inputs_hat.shape=[None* N, num_capsule, input_num_capsule, dim_capsule]
# 				# The first two dimensions as `batch` dimension,
# 				# then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
# 				# b.shape=[batch_size* N, num_capsule, input_num_capsule]
# 				b += K.batch_dot(outputs, inputs_hat_stopped, axes=[2,3])
# 		# End: Routing algorithm -----------------------------------------------------------------------#

# 		outputs = K.reshape(outputs, shape=tf.stack([batch_size, num_neighbours, 
# 			self.num_capsule, self.dim_capsule]) )

# 		return outputs

# 	def compute_output_shape(self, input_shape):
# 		# print "graphcaps shape", tuple(list(input_shape[:2]) + [self.num_capsule, self.dim_capsule])
# 		return tuple(list(input_shape[:2]) + [self.num_capsule, self.dim_capsule])

# class SimpleAggregateLayer(layers.Layer):

# 	def __init__(self, num_neighbours, mode="length", **kwargs):
# 		super(SimpleAggregateLayer, self).__init__(**kwargs)

# 		self.num_neighbours = num_neighbours
# 		self.mode = mode

# 	def build(self, input_shape):
# 		'''
# 		input_shape = [batch_size, Nn, input_dim]
# 		'''

# 		assert len(input_shape) == 4, "incorrect simple aggregation input"

# 		# print "input shape", input_shape
# 		# self.n_dimension = int(input_shape[1] // self.num_neighbours)
# 		# print "n dimension",self.n_dimension

# 		self.num_input_caps = input_shape[2]
# 		self.input_cap_dim = input_shape[3]
# 		# self.input_dim = input_shape[2]

# 		# initializer = initializers.get(self.kernel_initializer)
# 		# # regularizer = regularizers.get(self.kernel_regularizer)
# 		# regularizer = l2(self.kernel_regularizer)

# 		# self.W = self.add_weight(shape=(self.input_dim, self.num_caps * self.num_filters * self.new_dim), 
# 		#                           trainable=True, initializer=initializer, regularizer=regularizer,
# 		#                           name="W")

# 		# self.bias = self.add_weight(shape=(self.num_caps * self.num_filters * self.new_dim, ),
# 		#                           trainable=True, initializer=initializer, regularizer=regularizer,
# 		#                           name="bias")

# 		self.built = True

# 	def get_config(self):
# 		config = super(SimpleAggregateLayer, self).get_config()
# 		config.update({"num_neighbours":self.num_neighbours, "mode":self.mode,})
# 		return config

# 	def length(self, inputs, axis=-1):
# 		return K.sqrt(K.sum(K.square(inputs), axis=axis, keepdims=False) + K.epsilon())# + K.epsilon()

# 	def call(self, inputs):

# 		'''
# 		input shape = [None, Nn, num_inputs_caps, cap_dim]
# 		'''
		
# 		#aggregate over neighbours
# 		if self.mode == "mean":
# 			agg_fn = K.mean
# 		elif self.mode == "max":
# 			agg_fn = K.max
# 		elif self.mode == "length":
# 			agg_fn = self.length
# 		else:
# 			raise Exception

# 		inputs_shaped = K.reshape(inputs, shape=[-1, self.num_neighbours, 
# 			self.num_input_caps, self.input_cap_dim])

# 		output = agg_fn(inputs_shaped, axis=1)

# 		output = K.reshape(output, shape=tf.stack([K.shape(inputs)[0], -1, self.num_input_caps, self.input_cap_dim]))

# 		return output
		
# 	def compute_output_shape(self, input_shape):
# 		'''
# 		input_shape = [None, Nn, num_caps, cap_dim]
# 		output_shape is [None, Nn+1, num_caps, cap_dim]

# 		'''     
# 		return tuple([input_shape[0], int(input_shape[1] // self.num_neighbours), 
# 			self.num_input_caps, self.input_cap_dim])

# class AggregateLayer(layers.Layer):
# 	"""
# 	A layer to perform a convolutional filter on a graph with mean aggregation
# 	TODO
	
# 	Author: David McDonald, Email: `dxm237@cs.bham.ac.uk'
# 	"""
# 	def __init__(self, num_neighbours, num_caps, num_filters, new_dim, mode="mean", activation=None,
# 				 kernel_initializer='glorot_uniform', kernel_regularizer=reg,
# 				 **kwargs):
# 		super(AggregateLayer, self).__init__(**kwargs)

# 		self.num_neighbours = num_neighbours
# 		self.num_caps = num_caps
# 		self.num_filters = num_filters
# 		self.new_dim = new_dim
# 		self.mode = mode
# 		self.activation = activation
# 		self.kernel_initializer = kernel_initializer#RandomUniform(minval=-0.5, maxval=0.5)#kernel_initializer
# 		self.kernel_regularizer = kernel_regularizer

# 	def build(self, input_shape):
# 		'''
# 		input_shape = [batch_size, Nn, input_dim]
# 		'''
# 		assert len(input_shape) == 3, "incorrect aggregation input"

# 		# self.n_dimension = input_shape[1] 
# 		# self.num_input_caps = input_shape[2]
# 		# self.input_dim = input_shape[3]
# 		self.input_dim = input_shape[2]

# 		initializer = initializers.get(self.kernel_initializer)
# 		# regularizer = regularizers.get(self.kernel_regularizer)
# 		regularizer = l2(self.kernel_regularizer)

# 		self.W = self.add_weight(shape=(self.input_dim, self.num_caps * self.num_filters * self.new_dim), 
# 									trainable=True, initializer=initializer, regularizer=regularizer,
# 									name="W")

# 		self.bias = self.add_weight(shape=(self.num_caps * self.num_filters * self.new_dim, ),
# 									trainable=True, initializer=initializer, regularizer=regularizer,
# 									name="bias")

# 		self.built = True

# 	def get_config(self):
# 		config = super(AggregateLayer, self).get_config()
# 		config.update({"num_neighbours":self.num_neighbours, "num_caps": self.num_caps,
# 			"num_filters":self.num_filters, "new_dim":self.new_dim, "mode":self.mode, "activation":self.activation,
# 			"kernel_initializer":self.kernel_initializer, "kernel_regularizer":self.kernel_regularizer})
# 		return config

# 	def call(self, inputs):

# 		'''
# 		input shape = [None, Nn, num_inputs_caps, cap_dim]
# 		'''
		
# 		#aggregate over neighbours


# 		inputs_shaped = K.reshape(inputs, shape=[-1, self.num_neighbours, self.input_dim])
# 		# print inputs_shaped.shape
# 		# raise SystemExit
# 		# shape is now [batch*Nn+1, num_neighbours, input_dim]
# 		if self.mode == "mean":
# 			agg_fn = K.mean
# 		else:
# 			raise Exception
# 		inputs_aggregated = agg_fn(inputs_shaped, axis=1)
# 		# print K.eval(inputs_aggregated)
# 		# raise SystemExit
# 		# shape is now [batch*Nn+1, input_dim]
# 		output = K.dot(inputs_aggregated, self.W)# + self.bias
# 		# print output.shape
# 		output = K.bias_add(output, self.bias)
# 		# print output.shape
# 		# raise SystemExit
# 		# shape is not [batch*Nn+1, num_caps*num_filters*new_dim]


		
# 		# if self.activation is not None:
# 		#   output = activations.get(self.activation)(output)
# 		# output = squash(output)

# 		output = K.reshape(output, shape=tf.stack([K.shape(inputs)[0], -1, self.num_caps*self.num_filters, self.new_dim]))
# 		output = squash(output)
# 		# print output.shape
# 		# raise SystemExit
# 		return output
		
# 	def compute_output_shape(self, input_shape):
# 		'''
# 		input_shape = [None, Nn, num_caps, cap_dim]
# 		output_shape is [None, Nn+1, num_caps, cap_dim]

# 		'''     
# 		return tuple([input_shape[0], int(input_shape[1] // self.num_neighbours), 
# 			self.num_caps*self.num_filters, self.new_dim])

# class HyperbolicDistanceLayer(layers.Layer):
# 	"""
# 	TODO
# 	"""
# 	def __init__(self, num_positive_samples, num_negative_samples, **kwargs):
# 		super(HyperbolicDistanceLayer, self).__init__(**kwargs)
# 		self.num_positive_samples = num_positive_samples
# 		self.num_negative_samples = num_negative_samples

# 	def build(self, input_shape):
# 		assert input_shape is not None
# 		self.N = input_shape[1]
# 		self.step_size = int(self.N // (1 + self.num_positive_samples + self.num_negative_samples))
# 		self.built = True
# 		# print input_shape, self.N, self.num_positive_samples, self.num_negative_samples, self.step_size
# 		# print range(0, self.N, self.step_size)

# 		# print input_shape, self.N, self.step_size, range(0, self.N, self.step_size)

# 	def get_config(self):
# 		config = super(HyperbolicDistanceLayer, self).get_config()
# 		config.update({"num_positive_samples":self.num_positive_samples, 
# 					   "num_negative_samples":self.num_negative_samples})
# 		return config
		
	# def safe_norm(self, x, axis=-1, sqrt=False, clip=False):
	# 	y = K.sum(K.square(x), axis=axis, keepdims=False) + K.epsilon()
	# 	if sqrt:
	# 		y = K.sqrt(y)
	# 	if clip:
	# 		y = K.clip(y, min_value=K.epsilon(), max_value=1-K.epsilon())
	# 	return y

	# def squared_norm(self, x, axis=-1, sqrt=False, clip=False):
	# 	y = K.sum(K.square(x), axis=axis, keepdims=False)
	# 	# if sqrt:
	# 	# 	y = K.sqrt(y)
	# 	# if clip:
	# 	# 	y = K.clip(y, min_value=K.epsilon(), max_value=1-K.epsilon())
	# 	return y
		
	# def call(self, inputs):
	# 	'''
	# 	input_shape = [None, N, D]
	# 	'''

	# 	def acosh(x):
	# 		return K.log(x + K.sqrt(K.square(x) - 1))

	# 	inputs = inputs[:,0:self.N:self.step_size]
	# 	# print inputs.shape, K.shape(inputs)[1], self.step_size
	# 	# raise SystemExit
	# 	u = inputs[:,:1]
	# 	v = inputs[:,1:]

	# 	num = self.squared_norm(u - v)
	# 	nu = self.squared_norm(u)
	# 	nv = self.squared_norm(v)

	# 	num = tf.clip_by_value(num, min_norm, max_)
	# 	nu = tf.clip_by_value(nu, min_norm, max_norm)
	# 	nv = tf.clip_by_value(nv, min_norm, max_norm)
	# 	# nu = tf.clip_by_value(nu, np.sqrt(1-max_norm), max_norm)
	# 	# nv = tf.clip_by_value(nv, np.sqrt(1-max_norm), max_norm)

	# 	den = (1 - nu) * (1 - nv) 

	# 	d = num / den
	# 	d = tf.clip_by_value(d, min_norm, 
	# 		np.nextafter(np.sqrt(max_), 0, dtype=K.floatx()) / 2)

	# 	return acosh(1 + 2 * d)
	# 	# return K.clip(tf.sqrt(num), min_value=min_norm, max_value=max_)

	# 	# return K.sqrt(num)

	# def compute_output_shape(self, input_shape):
	# 	# print "hyp dist", tuple([input_shape[0], self.num_positive_samples + self.num_negative_samples])
	# 	return tuple([input_shape[0], self.num_positive_samples + self.num_negative_samples])