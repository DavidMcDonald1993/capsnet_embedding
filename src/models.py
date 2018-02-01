
import numpy as np

from keras import layers
from keras.models import Model
from keras.regularizers import l2

from graphcaps_layers import AggregateLayer, GraphCapsuleLayer, HyperbolicDistanceLayer, Length, squash
from losses import masked_crossentropy, masked_margin_loss, hyperbolic_negative_sampling_loss

def build_graphcaps(data_dim, num_classes, embedding_dim, num_positive_samples, num_negative_samples, sample_sizes):

	output_size = 1 + num_positive_samples + num_negative_samples
	capsule_dim_first_layer = 4

	x = layers.Input(shape=(np.prod(sample_sizes + 1) * output_size, 1, data_dim), name="input_signal")
	# x_norm = layers.BatchNormalization()(x)
	agg1 = AggregateLayer(num_neighbours=sample_sizes[-1] + 1, num_filters=8, new_dim=128,
								activation=None, name="agg1")(x)
	cap1 = GraphCapsuleLayer(num_capsule=num_classes, dim_capsule=capsule_dim_first_layer, num_routing=3, name="cap1")(agg1)



	cap1_squash = layers.Lambda(squash, output_shape=lambda x: x, name="cap1_squash")(cap1)
	label_predictions = Length(name="label_predictions")(cap1_squash)



	# cap1_length = Length(name="length_first_layer")(cap1)
	embedding_first_layer = layers.Reshape((-1, num_classes*capsule_dim_first_layer))(cap1)
	embedding_first_layer = layers.Lambda(squash, output_shape=lambda x : x, name="embedding_first_layer")(embedding_first_layer)
	hyperbolic_distance_first_layer = HyperbolicDistanceLayer(num_positive_samples=num_positive_samples,
		num_negative_samples=num_negative_samples, name="hyperbolic_distance_first_layer")(embedding_first_layer)







	agg2 = AggregateLayer(num_neighbours=sample_sizes[-2] + 1, num_filters=16, new_dim=256,
								activation=None, name="agg2")(cap1_squash)
	cap2 = GraphCapsuleLayer(num_capsule=1, dim_capsule=embedding_dim, num_routing=1, name="cap2")(agg2)

	# cap2_length = Length(name="cap2_length")(cap2)
	# label_predictions = Length(name="label_predictions")(cap2)

	embedding_second_layer = layers.Reshape((-1, embedding_dim))(cap2)
	# cap2_squash = layers.Lambda(squash, output_shape=lambda x: x)(cap2)
	# cap2_length = Length(name="length_second_layer")(cap2)
	embedding_second_layer = layers.Lambda(squash, output_shape=lambda x : x, name="embedding_second_layer")(embedding_second_layer)
	hyperbolic_distance_second_layer = HyperbolicDistanceLayer(num_positive_samples=num_positive_samples,
		num_negative_samples=num_negative_samples, name="hyperbolic_distance_second_layer")(embedding_second_layer)




	# agg3 = AggregateLayer(num_neighbours=sample_sizes[-3] + 1, num_filters=8, new_dim=16,
	# 							activation=None, name="agg3")(cap2)
	# cap3 = GraphCapsuleLayer(num_capsule=embedding_dim, dim_capsule=8, num_routing=1, name="cap3")(agg3)

	# cap3_length = Length(name="cap3_length")(cap3)
	# embedding_third_layer = layers.Lambda(squash, output_shape=lambda x : x, name="embedding_third_layer")(cap3_length)
	# hyperbolic_distance_third_layer = HyperbolicDistanceLayer(num_positive_samples=num_positive_samples,
	# 	num_negative_samples=num_negative_samples, name="hyperbolic_distance_third_layer")(embedding_third_layer)

	

	model = Model(x, #[hyperbolic_distance_first_layer])#, hyperbolic_distance_second_layer])
		[label_predictions, hyperbolic_distance_first_layer, hyperbolic_distance_second_layer]) 
	model.compile(optimizer="adam", 
		# loss=[hyperbolic_negative_sampling_loss]*1)
		loss=[masked_crossentropy, hyperbolic_negative_sampling_loss, hyperbolic_negative_sampling_loss],
		# hyperbolic_negative_sampling_loss, hyperbolic_negative_sampling_loss], 
		loss_weights=[0.0, 1.0, 1.0])

	embedder = Model(x, embedding_second_layer)

	return model, embedder