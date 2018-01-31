
import numpy as np

from keras import layers
from keras.models import Model

from graphcaps_layers import AggregateLayer, GraphCapsuleLayer, HyperbolicDistanceLayer, Length, squash
from losses import masked_crossentropy, masked_margin_loss, hyperbolic_negative_sampling_loss

def build_graphcaps(data_dim, num_classes, embedding_dim, num_positive_samples, num_negative_samples, sample_sizes):

	output_size = 1 + num_positive_samples + num_negative_samples

	x = layers.Input(shape=(np.prod(sample_sizes + 1) * output_size, 1, data_dim), name="input_signal")
	# x_norm = layers.BatchNormalization()(x)
	agg1 = AggregateLayer(num_neighbours=sample_sizes[-1] + 1, num_filters=8, new_dim=128,
								activation=None, name="agg1")(x)
	cap1 = GraphCapsuleLayer(num_capsule=4, dim_capsule=4, num_routing=3, name="cap1")(agg1)

	# cap1_length = Length(name="cap1_length")(cap1)
	# cap1 = layers.Lambda(squash, output_shape=lambda x: x)(cap1)

	cap1_length = layers.Reshape((-1, 4*4))(cap1)
	cap1_length = layers.Lambda(squash)(cap1_length)
	
	# embedding_first_layer = layers.Lambda(squash, output_shape=lambda x : x, name="embedding_first_layer")(cap1_length)
	embedding_first_layer = cap1_length
	hyperbolic_distance_first_layer = HyperbolicDistanceLayer(num_positive_samples=num_positive_samples,
		num_negative_samples=num_negative_samples, name="hyperbolic_distance_first_layer")(embedding_first_layer)

	agg2 = AggregateLayer(num_neighbours=sample_sizes[-2] + 1, num_filters=16, new_dim=256,
								activation=None, name="agg2")(cap1)
	cap2 = GraphCapsuleLayer(num_capsule=1, dim_capsule=embedding_dim, num_routing=1, name="cap2")(agg2)

	# cap2_length = Length(name="cap2_length")(cap2)
	cap2 = layers.Lambda(squash, output_shape=lambda x: x)(cap2)
	# label_predictions = Length(name="label_predictions")(cap2)

	cap2_length = layers.Reshape((-1, embedding_dim))(cap2)

	# embedding_second_layer = layers.Lambda(squash, output_shape=lambda x : x, name="embedding_second_layer")(cap2_length)
	embedding_second_layer = cap2_length
	hyperbolic_distance_second_layer = HyperbolicDistanceLayer(num_positive_samples=num_positive_samples,
		num_negative_samples=num_negative_samples, name="hyperbolic_distance_second_layer")(embedding_second_layer)




	# agg3 = AggregateLayer(num_neighbours=sample_sizes[-3] + 1, num_filters=8, new_dim=16,
	# 							activation=None, name="agg3")(cap2)
	# cap3 = GraphCapsuleLayer(num_capsule=embedding_dim, dim_capsule=8, num_routing=1, name="cap3")(agg3)

	# cap3_length = Length(name="cap3_length")(cap3)
	# embedding_third_layer = layers.Lambda(squash, output_shape=lambda x : x, name="embedding_third_layer")(cap3_length)
	# hyperbolic_distance_third_layer = HyperbolicDistanceLayer(num_positive_samples=num_positive_samples,
	# 	num_negative_samples=num_negative_samples, name="hyperbolic_distance_third_layer")(embedding_third_layer)

	

	model = Model(x, [hyperbolic_distance_first_layer, hyperbolic_distance_second_layer])
		# [label_predictions, hyperbolic_distance_first_layer, hyperbolic_distance_second_layer, hyperbolic_distance_third_layer]) 
	model.compile(optimizer="adam", 
		loss=[hyperbolic_negative_sampling_loss]*2)
		# loss=[masked_margin_loss, hyperbolic_negative_sampling_loss, ])
		# hyperbolic_negative_sampling_loss, hyperbolic_negative_sampling_loss], 
		# loss_weights=[0.0, 1.0, 0.0, 0.0])

	embedder = Model(x, embedding_second_layer)

	return model, embedder