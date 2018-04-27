
import os
import re
import numpy as np

# import tensorflow as tf
from keras import layers
from keras.models import Model, load_model
# from keras.regularizers import l2
from keras.optimizers import Adam
# from keras import backend as K

from graphcaps_layers import SimpleAggregateLayer, AggregateLayer, AggGraphCapsuleLayer, GraphCapsuleLayer, HyperbolicDistanceLayer, Length, squash, inverse_squash
from losses import masked_crossentropy, masked_margin_loss, hyperbolic_negative_sampling_loss

def connect_layers(layer_tuples, x):
	
	y = x

	for layer_tuple in layer_tuples:
		for layer in layer_tuple:
			y = layer(y)

	return y

# def load_best_model(model_path):

# 	model = load_model(os.path.join(model_path, "best_model.h5"),
# 		custom_objects={"AggregateLayer":AggregateLayer, 
# 						"Length":Length,
# 						 "HyperbolicDistanceLayer":HyperbolicDistanceLayer, 
# 						 "GraphCapsuleLayer": GraphCapsuleLayer,
# 						 "hyperbolic_negative_sampling_loss": hyperbolic_negative_sampling_loss, 
# 						 "masked_margin_loss": masked_margin_loss,
# 						 "masked_crossentropy": masked_crossentropy})
# 	return model


def load_models(X, Y, model_path, args, load_best=False):

	saved_models = sorted([f for f in os.listdir(model_path) if re.match(r"^[0-9][0-9][0-9][0-9]*", f)])
	initial_epoch = len(saved_models)

	data_dim = X.shape[1]
	num_classes = Y.shape[1]

	if initial_epoch == 0:

		print ("Creating new model")

		model = generate_graphcaps_model(data_dim, num_classes, args)

	else:

		if load_best:
			model_file = os.path.join(model_path, "best_model.h5")
			print ("loading best model")
		else:
			model_file = os.path.join(model_path, saved_models[-1])
			print ("Loading model from file")


		model = load_model(model_file,
			custom_objects={"squash":squash,
							"SimpleAggregateLayer": SimpleAggregateLayer,
							"AggregateLayer":AggregateLayer, 
							"Length":Length,
							 "HyperbolicDistanceLayer":HyperbolicDistanceLayer, 
							 "AggGraphCapsuleLayer": AggGraphCapsuleLayer,
							 "GraphCapsuleLayer": GraphCapsuleLayer,
							 "hyperbolic_negative_sampling_loss": hyperbolic_negative_sampling_loss, 
							 "masked_margin_loss": masked_margin_loss,
							 "masked_crossentropy": masked_crossentropy})


	model.summary()
	# raise SystemExit

	embedder, label_prediction_model = build_embedder_and_prediction_model(data_dim, num_classes, model, args)

	# add 1 to initial epoch to account for epoch zero showing losses before any training happens
	# initial_epoch += 1

	return model, embedder, label_prediction_model, initial_epoch

def build_embedder_and_prediction_model(data_dim, num_classes, model, args):

	neighbourhood_sample_sizes = args.neighbourhood_sample_sizes
	number_of_capsules_per_layer = args.number_of_capsules_per_layer

	def model_to_dict(model):
		layer_dict = {}
		for layer in sorted(model.layers, key=lambda l: l.name):
			layer_name = layer.name.split("_layer")[0]
			layer_dict.setdefault(layer_name, []).append(layer)
		return layer_dict

	layer_dict = model_to_dict(model)
	# for d in layer_dict:
	# 	for l in layer_dict[d]:
	# 		print l.name
	# raise SystemExit
	num_layers = len(neighbourhood_sample_sizes)

	'''
	BUILD EMBEDDER MODEL FROM DICTIONARY OF LAYERS

	'''
	print()
	print ("EMBEDDER")
	embedding_layer = num_layers
	layer_tuple_list = list(zip(\
		# layer_dict["input_normalization"][:embedding_layer], 
		# layer_dict["aggregation"][:embedding_layer],
		# layer_dict["batch_normalization"][:embedding_layer],
		# layer_dict["cap_input"][:embedding_layer], 
		layer_dict["cap"][:embedding_layer], )) + [(layer_dict["feature_prob"][-1], layer_dict["embedding"][-1], )]
		# layer_dict["feature_prob"][:embedding_layer])) +\
	# [(layer_dict["embedding"][-1], )]

	for t in layer_tuple_list:
		for layer in t:
			print layer.name
	# embedder_lambda = lambda x: connect_layers(layer_tuple_list, x)

	embedder_input_num_neighbours = np.prod(neighbourhood_sample_sizes + 1)
	embedder_input = layers.Input(shape=(embedder_input_num_neighbours, 1, data_dim), name="embedder_input")
	embedder_output = connect_layers(layer_tuple_list, embedder_input)

	embedder = Model(embedder_input, embedder_output)
	embedder.summary()
	# raise SystemExit

	'''

	BUILD LABEL PREDICTOR MODEL FROM DICTIONARY OF LAYERS
	'''
	print()
	print ("PREDICTOR")
	label_prediction_layers = np.where(number_of_capsules_per_layer==num_classes)[0] + 1
	if num_classes > 1 and len(label_prediction_layers) > 0:
		
		label_prediction_layer = label_prediction_layers[-1]

		label_prediction_input_num_neighbours = np.prod(neighbourhood_sample_sizes[:label_prediction_layer] + 1)
		label_prediction_input = layers.Input(shape=(label_prediction_input_num_neighbours, 1, data_dim),
			name="label_prediction_input")

		layer_tuple_list = list(zip(\
			# layer_dict["input_normalization"][:label_prediction_layer], 
			# layer_dict["aggregation"][:label_prediction_layer],
			# layer_dict["batch_normalization"][:label_prediction_layer], 
			# layer_dict["cap_input"][:label_prediction_layer], 
			layer_dict["cap"][:label_prediction_layer], )) + [(layer_dict["feature_prob"][-1], )]
			# layer_dict["feature_prob"][:label_prediction_layer]))

		for t in layer_tuple_list:
			for layer in t:
				print layer.name

		label_prediction_output = connect_layers(layer_tuple_list, label_prediction_input)

		label_prediction_model = Model(label_prediction_input, label_prediction_output)
		label_prediction_model.summary()
		# raise SystemExit
	else: 
		label_prediction_model = None
	print()
	# raise SystemExit

	return embedder, label_prediction_model

def generate_graphcaps_model(data_dim, num_classes, args):

	neighbourhood_sample_sizes = args.neighbourhood_sample_sizes
	num_primary_caps_per_layer = args.num_primary_caps_per_layer
	num_filters_per_layer = args.num_filters_per_layer
	agg_dim_per_layer = args.agg_dim_per_layer
	number_of_capsules_per_layer = args.number_of_capsules_per_layer
	capsule_dim_per_layer = args.capsule_dim_per_layer

	num_positive_samples = args.num_positive_samples
	num_negative_samples = args.num_negative_samples


	num_layers = len(neighbourhood_sample_sizes)
	output_size = 1 + num_positive_samples + num_negative_samples
	num_neighbours_per_layer = np.array([np.prod(neighbourhood_sample_sizes[i:]+1) * output_size for i in range(num_layers+1)])
	# number_of_capsules_per_layer = np.append(1, number_of_capsules_per_layer)


	x = layers.Input(shape=(num_neighbours_per_layer[0], 1, data_dim), 
		name="input_layer")
	y = x

	# embeddings = []
	hyperbolic_distances = []
	label_predictions = []


	for i, neighbourhood_sample_size, num_primary_caps, num_filters, agg_dim, num_caps, capsule_dim in zip(range(num_layers),
		neighbourhood_sample_sizes, num_primary_caps_per_layer, num_filters_per_layer, agg_dim_per_layer, 
		number_of_capsules_per_layer, capsule_dim_per_layer):

		# y = layers.BatchNormalization(name="input_normalization_layer_{}".format(i))(y)
		# y = SimpleAggregateLayer(num_neighbours=neighbourhood_sample_size+1,
		# 	name="aggregation_layer_{}".format(i))(y)
		# y = AggregateLayer(num_neighbours=neighbourhood_sample_size+1, num_caps=num_primary_caps,
		# 	num_filters=num_filters, new_dim=agg_dim,
		# 	activation="relu", name="aggregation_layer_{}".format(i))(y)
		# y = layers.BatchNormalization(name="batch_normalization_layer_{}".format(i))(y)
		# y = layers.Reshape([-1, num_primary_caps, num_filters*agg_dim], name="cap_input_layer_{}".format(i))(y)
		
		if num_caps == 1:
			num_routing = 1
		else:
			num_routing = 3 

		# y = layers.Lambda(squash, name="cap_layer_{}".format(i))(y)
		# y = GraphCapsuleLayer(num_capsule=num_caps, dim_capsule=capsule_dim, num_routing=num_routing, 
		# 	name="cap_layer_{}".format(i))(y)
		y = AggGraphCapsuleLayer(num_neighbours=neighbourhood_sample_size+1,
			num_capsule=num_caps, dim_capsule=capsule_dim, num_routing=num_routing, 
			name="cap_layer_{}".format(i))(y)
		# print "y output", y.get_shape()
		feature_prob = Length(name="feature_prob_layer_{}".format(i))(y)
		# print "feat output", feature_prob.get_shape()
		if num_caps == num_classes:
			label_predictions.append(feature_prob)

		layer_embedding = layers.Lambda(inverse_squash, name="embedding_layer_{}".format(i))(feature_prob)
		# print "emb output", layer_embedding.get_shape()
		assert layer_embedding.get_shape() is not None
		layer_hyperbolic_distance = HyperbolicDistanceLayer(num_positive_samples=num_positive_samples,
			num_negative_samples=num_negative_samples, 
			name="hyperbolic_distance_layer_{}".format(i))(layer_embedding)

		# layer_embedding = layers.Reshape([-1, num_caps*capsule_dim], name="embedding_reshape_layer_{}".format(i))(y)
		# layer_embedding = layers.Lambda(squash, name="embedding_squash_layer_{}".format(i))(layer_embedding)
		# layer_hyperbolic_distance = HyperbolicDistanceLayer(num_positive_samples=num_positive_samples,
		# 	num_negative_samples=num_negative_samples, name="hyperbolic_distance_layer_{}".format(i))(layer_embedding)

		# embeddings.append(layer_embedding)
		hyperbolic_distances.append(layer_hyperbolic_distance)

		# y = layers.Lambda(squash, name="squash_layer_{}".format(i))(y)

		# if num_caps == num_classes:
		# 	label_prediction = Length(name="label_prediction_layer_{}".format(i))(y)
		# 	label_predictions.append(label_prediction)

	losses = [masked_margin_loss]*len(label_predictions) +\
	[hyperbolic_negative_sampling_loss]*len(hyperbolic_distances)
	
	loss_weights = []
	if args.use_labels:
		loss_weights += [1./len(label_predictions)] * len(label_predictions)
	else:
		loss_weights += [0.] * len(label_predictions)
	if args.no_embedding_loss:
		loss_weights += [0.] * len(hyperbolic_distances)
	elif args.no_intermediary_loss:
		loss_weights += [0.] * (len(hyperbolic_distances) - 1)  + [1e-0]
	else:
		# loss_weights += [1e-0/len(hyperbolic_distances)]*len(hyperbolic_distances)
		loss_weights += [1e-0]*len(hyperbolic_distances)

	# loss_weights = [1e-3, 1, 1e-3]
	print ("generating model with loss weights:", loss_weights)

	graphcaps = Model(x,  label_predictions + hyperbolic_distances)
	# adam = Adam(lr=1e-4, clipnorm=1)
	# adam = Adam(lr=1e-4)
	graphcaps.compile(optimizer="adam", loss=losses, loss_weights=loss_weights)
	# raise SystemExit
	return graphcaps
