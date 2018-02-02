
import numpy as np

from keras import layers
from keras.models import Model
from keras.regularizers import l2

from graphcaps_layers import AggregateLayer, GraphCapsuleLayer, HyperbolicDistanceLayer, Length, squash
from losses import masked_crossentropy, masked_margin_loss, hyperbolic_negative_sampling_loss

def generate_graphcaps_model(data_dim, num_classes, num_positive_samples, num_negative_samples,
	neighbourhood_sample_sizes, num_filters_per_layer, agg_dim_per_layer,
	number_of_capsules_per_layer, capsule_dim_per_layer):

	output_size = 1 + num_positive_samples + num_negative_samples

	x = layers.Input(shape=(np.prod(neighbourhood_sample_sizes + 1) * output_size, 1, data_dim), name="input_layer")

	y = x

	label_predictions = []
	embeddings = []
	hyperbolic_distances = []

	num_layers = len(neighbourhood_sample_sizes)

	for i, neighbourhood_sample_size, num_filters, agg_dim, num_caps, capsule_dim in zip(range(num_layers),
		neighbourhood_sample_sizes, number_of_capsules_per_layer, agg_dim_per_layer, 
		number_of_capsules_per_layer, capsule_dim_per_layer):

		if num_caps == 1:
			num_routing = 1
		else:
			num_routing = 3 

		y = AggregateLayer(num_neighbours=neighbourhood_sample_size+1, num_filters=num_filters, new_dim=agg_dim,
			activation="relu", name="agg_layer_{}".format(i))(y)
		y = GraphCapsuleLayer(num_capsule=num_caps, dim_capsule=capsule_dim, num_routing=num_routing, 
			name="cap_layer_{}".format(i))(y)

		layer_embedding = layers.Reshape((-1, num_caps*capsule_dim), name="embedding_reshape_layer_{}".format(i))(y)
		layer_embedding = layers.Lambda(squash, name="embedding_squash_layer_{}".format(i))(layer_embedding)
		layer_hyperbolic_distance = HyperbolicDistanceLayer(num_positive_samples=num_positive_samples,
			num_negative_samples=num_negative_samples, name="hyperbolic_distance_layer_{}".format(i))(layer_embedding)

		embeddings.append(layer_embedding)
		hyperbolic_distances.append(layer_hyperbolic_distance)

		y = layers.Lambda(squash, name="squash_layer_{}".format(i))(y)

		if num_caps == num_classes:
			label_prediction = Length(name="label_prediction_layer_{}".format(i))(y)
			label_predictions.append(label_prediction)


	graphcaps = Model(x, label_predictions + hyperbolic_distances)
	graphcaps.compile(optimizer="adam", 
		loss=[masked_crossentropy]*len(label_predictions) + [hyperbolic_negative_sampling_loss]*len(embeddings), 
		loss_weights=[0]*len(label_predictions) + [1./len(hyperbolic_distances)]*len(hyperbolic_distances))

	embedder = Model(x, embeddings[-1])

	return graphcaps, embedder

def build_graphcaps(data_dim, num_classes, embedding_dim, num_positive_samples, num_negative_samples, sample_sizes):

	output_size = 1 + num_positive_samples + num_negative_samples
	
	num_capsules_first_layer = 16
	capsule_dim_first_layer = 8

	num_capsules_second_layer = num_classes
	capsule_dim_second_layer = 4

	num_capsules_third_layer = 1
	capsule_dim_third_layer = embedding_dim

	x = layers.Input(shape=(np.prod(sample_sizes + 1) * output_size, 1, data_dim), name="input_signal")
	# x_norm = layers.BatchNormalization()(x)
	agg1 = AggregateLayer(num_neighbours=sample_sizes[-1] + 1, num_filters=16, new_dim=128,
								activation="relu", name="agg1")(x)
	cap1 = GraphCapsuleLayer(num_capsule=num_capsules_first_layer, dim_capsule=capsule_dim_first_layer, 
		num_routing=3, name="cap1")(agg1)



	cap1_squash = layers.Lambda(squash, output_shape=lambda x: x, name="cap1_squash")(cap1)
	# label_predictions = Length(name="label_predictions")(cap1_squash)



	# cap1_length = Length(name="length_first_layer")(cap1)
	embedding_first_layer = layers.Reshape((-1, num_capsules_first_layer*capsule_dim_first_layer),
		name="reshape_first_layer")(cap1)
	embedding_first_layer = layers.Lambda(squash, output_shape=lambda x : x, name="embedding_first_layer")(embedding_first_layer)
	hyperbolic_distance_first_layer = HyperbolicDistanceLayer(num_positive_samples=num_positive_samples,
		num_negative_samples=num_negative_samples, name="hyperbolic_distance_first_layer")(embedding_first_layer)







	agg2 = AggregateLayer(num_neighbours=sample_sizes[-2] + 1, num_filters=16, new_dim=32,
								activation="relu", name="agg2")(cap1_squash)
	cap2 = GraphCapsuleLayer(num_capsule=num_capsules_second_layer, dim_capsule=capsule_dim_second_layer, 
		num_routing=3, name="cap2")(agg2)

	# cap2_length = Length(name="cap2_length")(cap2)
	cap2_squash = layers.Lambda(squash, output_shape=lambda x: x)(cap2)
	label_predictions = Length(name="label_predictions")(cap2_squash)

	embedding_second_layer = layers.Reshape((-1, num_capsules_second_layer*capsule_dim_second_layer),
		name="reshape_second_layer")(cap2)
	embedding_second_layer = layers.Lambda(squash, output_shape=lambda x : x, name="embedding_second_layer")(embedding_second_layer)
	hyperbolic_distance_second_layer = HyperbolicDistanceLayer(num_positive_samples=num_positive_samples,
		num_negative_samples=num_negative_samples, name="hyperbolic_distance_second_layer")(embedding_second_layer)




	agg3 = AggregateLayer(num_neighbours=sample_sizes[-3] + 1, num_filters=8, new_dim=16,
								activation="relu", name="agg3")(cap2_squash)
	cap3 = GraphCapsuleLayer(num_capsule=num_capsules_third_layer, dim_capsule=capsule_dim_third_layer, 
		num_routing=1, name="cap3")(agg3)

	# cap3_length = Length(name="cap3_length")(cap3)
	embedding_third_layer = layers.Reshape((-1, num_capsules_third_layer*capsule_dim_third_layer),
		name="reshape_third_layer")(cap3)
	embedding_third_layer = layers.Lambda(squash, output_shape=lambda x : x, name="embedding_third_layer")(embedding_third_layer)
	hyperbolic_distance_third_layer = HyperbolicDistanceLayer(num_positive_samples=num_positive_samples,
		num_negative_samples=num_negative_samples, name="hyperbolic_distance_third_layer")(embedding_third_layer)

	

	model = Model(x, #[hyperbolic_distance_first_layer])#, hyperbolic_distance_second_layer])
		[label_predictions, label_predictions, 
		hyperbolic_distance_first_layer, hyperbolic_distance_second_layer, hyperbolic_distance_third_layer]) 
	model.compile(optimizer="adam", 
		# loss=[hyperbolic_negative_sampling_loss]*1)
		loss=[masked_crossentropy] * 2 + [hyperbolic_negative_sampling_loss] * 3,
		# hyperbolic_negative_sampling_loss, hyperbolic_negative_sampling_loss], 
		loss_weights=[0.0, 0.0, 0.333, 0.33333, 0.333333])

	embedder = Model(x, embedding_third_layer)

	return model, embedder