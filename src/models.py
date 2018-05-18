
import os
import re
import numpy as np

import tensorflow as tf
from keras import layers
from keras.models import Model, load_model
from keras.regularizers import l1, l2
from keras.optimizers import Adam
from keras.initializers import RandomUniform, Constant
import keras.backend as K

from graphcaps_layers import UHatLayer, NeighbourhoodSamplingLayer, DynamicRoutingLayer, Mask
from graphcaps_layers import Length, squash, embedding_function
# from graphcaps_layers import SimpleAggregateLayer, AggregateLayer, AggGraphCapsuleLayer, GraphCapsuleLayer, HyperbolicDistanceLayer, Length, squash, embedding_function
from losses import masked_margin_loss, hyperbolic_negative_sampling_loss

# def connect_layers(layer_list, x):
	
# 	y = x

# 	for layer in layer_list:
# 		y = layer(y)

# 	return y


def load_graphcaps(X, Y, model_path, args, load_best=False):

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
							"Length":Length,
							"UHatLayer": UHatLayer,
							"NeighbourhoodSamplingLayer": NeighbourhoodSamplingLayer,
							"DynamicRoutingLayer": DynamicRoutingLayer,
							"Mask": Mask,
							"hyperbolic_negative_sampling_loss": hyperbolic_negative_sampling_loss, 
							"masked_margin_loss": masked_margin_loss,
							"tf":tf,
							"min_norm": 1e-7,
							"max_norm":np.nextafter(1, 0, dtype=K.floatx()),
							"max_": np.finfo(K.floatx()).max})
		# K.set_value(model.optimizer.lr, 1e-4)
		# print K.get_value(model.optimizer.lr)
		# raise SystemExit


	model.summary()
	# for l in model.output_layers:
		# print l.name
	# raise SystemExit
	# embedder, label_prediction_model = build_embedder_and_prediction_model(data_dim, num_classes, model, args)

	# add 1 to initial epoch to account for epoch zero showing losses before any training happens
	# initial_epoch += 1

	return model, initial_epoch
	# return model, embedder, label_prediction_model, initial_epoch

# def build_embedder_and_prediction_model(data_dim, num_classes, model, args):

# 	neighbourhood_sample_sizes = args.neighbourhood_sample_sizes
# 	number_of_capsules_per_layer = args.number_of_capsules_per_layer

# 	def model_to_dict(model):
# 		layer_dict = {}
# 		for layer in sorted(model.layers, key=lambda l: l.name):
# 			layer_name = layer.name.split("_layer")[0]
# 			layer_dict.setdefault(layer_name, []).append(layer)
# 		return layer_dict

# 	layer_dict = model_to_dict(model)
# 	num_layers = len(neighbourhood_sample_sizes)

# 	'''
# 	BUILD EMBEDDER MODEL FROM DICTIONARY OF LAYERS

# 	'''
# 	print()
# 	print ("EMBEDDER")
# 	embedding_layer = num_layers
# 	layer_list = []
# 	if args.num_primary_caps is not None:
# 		layer_list += [layer_dict["primary_cap"][0], layer_dict["primary_reshape"][0], layer_dict["primary_squash"][0]]
	
# 	layer_list += layer_dict["cap"][:embedding_layer]
# 	layer_list += [layer_dict["feature_prob"][-1], layer_dict["embedding_normalization"][-1], layer_dict["embedding"][-1]]

# 	# layer_list += [j for i in zip(layer_dict["cap"][:embedding_layer-1], layer_dict["squash"][:embedding_layer-1]) for j in i]
# 	# layer_list += [layer_dict["cap"][-1]]
# 	# layer_list += [layer_dict["embedding"][-1], layer_dict["embedding_squash"][-1]]

# 	embedder_input_num_neighbours = np.prod(neighbourhood_sample_sizes + 1)
# 	embedder_input = layers.Input(shape=(embedder_input_num_neighbours, 1, data_dim), name="embedder_input")
# 	embedder_output = connect_layers(layer_list, embedder_input)

# 	embedder = Model(embedder_input, embedder_output)
# 	embedder.summary()

# 	'''

# 	BUILD LABEL PREDICTOR MODEL FROM DICTIONARY OF LAYERS
# 	'''
# 	print()
# 	print ("PREDICTOR")
# 	label_prediction_layers = np.where(number_of_capsules_per_layer==num_classes)[0] + 1
# 	if num_classes > 1 and len(label_prediction_layers) > 0:
		
# 		label_prediction_layer = label_prediction_layers[-1]

# 		label_prediction_input_num_neighbours = np.prod(neighbourhood_sample_sizes[:label_prediction_layer] + 1)
# 		label_prediction_input = layers.Input(shape=(label_prediction_input_num_neighbours, 1, data_dim),
# 			name="label_prediction_input")

# 		layer_list = []
# 		if args.num_primary_caps is not None:
# 			layer_list += [layer_dict["primary_cap"][0], layer_dict["primary_reshape"][0], layer_dict["primary_squash"][0]]
		
# 		layer_list += layer_dict["cap"][:label_prediction_layer] 
# 		# layer_list += [j for i in zip(layer_dict["cap"][:label_prediction_layer], layer_dict["squash"][:label_prediction_layer]) for j in i]
# 		layer_list += [layer_dict["feature_prob"][-1]]

# 		label_prediction_output = connect_layers(layer_list, label_prediction_input)

# 		label_prediction_model = Model(label_prediction_input, label_prediction_output)
# 		label_prediction_model.summary()
# 	else: 
# 		label_prediction_model = None
# 	print()

# 	return embedder, label_prediction_model

def generate_graphcaps_model(data_dim, num_classes, args):

	neighbourhood_sample_sizes = args.neighbourhood_sample_sizes
	# num_primary_caps_per_layer = args.num_primary_caps_per_layer
	# num_filters_per_layer = args.num_filters_per_layer
	# agg_dim_per_layer = args.agg_dim_per_layer
	number_of_capsules_per_layer = args.number_of_capsules_per_layer
	capsule_dim_per_layer = args.capsule_dim_per_layer

	num_positive_samples = args.num_positive_samples
	num_negative_samples = args.num_negative_samples

	num_layers = len(neighbourhood_sample_sizes)
	output_size = 1 + num_positive_samples + num_negative_samples
	num_neighbours_per_layer = np.array([np.prod(neighbourhood_sample_sizes[i:]+1) * output_size for i in range(num_layers+1)])

	# N = len(G)

	# neighbours = [G.neighbors(n) for n in G.nodes()]

	x = layers.Input(shape=(data_dim, ))
	# x = layers.Input(shape=(num_neighbours_per_layer[0], 1, data_dim), name="input_layer")
	y = x

	adj_input = layers.Input(shape=(None,))

	if args.num_primary_caps is not None:
		y = layers.Dense(args.num_primary_caps * args.primary_cap_dim, activation=None, 
			kernel_regularizer=l2(1e-30), name="primary_cap_layer")(y)
		y = layers.Reshape([args.num_primary_caps, args.primary_cap_dim], name="primary_reshape_layer")(y)
		y = layers.Lambda(squash, output_shape=lambda x: x, 
			name="primary_squash_layer")(y)
	else:
		raise Exception
	embeddings = []
	# hyperbolic_distances = []
	label_predictions = []


	for i, neighbourhood_sample_size, num_caps, capsule_dim in zip(range(num_layers),
		neighbourhood_sample_sizes, number_of_capsules_per_layer, capsule_dim_per_layer):

		# y = layers.BatchNormalization(name="input_normalization_layer_{}".format(i))(y)
		# y = SimpleAggregateLayer(num_neighbours=neighbourhood_sample_size+1,
		# 	name="aggregation_layer_{}".format(i))(y)
		# y = AggregateLayer(num_neighbours=neighbourhood_sample_size+1, num_caps=num_primary_caps,
		# 	num_filters=num_filters, new_dim=agg_dim,
		# 	activation="relu", name="aggregation_layer_{}".format(i))(y)
		# y = layers.BatchNormalization(name="batch_normalization_layer_{}".format(i))(y)
		# y = layers.Reshape([-1, num_primary_caps, num_filters*agg_dim], name="cap_input_layer_{}".format(i))(y)
		
		# if num_caps == 1:
		# 	num_routing = 1
		# else:
		# 	num_routing = 3

		# print y.shape
		y = UHatLayer(num_capsule=num_caps, dim_capsule=capsule_dim, 
			name="u_hat_layer_{}".format(i))(y)
		# print "otsidem ", y.shape
		y = NeighbourhoodSamplingLayer(sample_size=neighbourhood_sample_size,
			name="neighbourhood_sample_layer_{}".format(i))([y, adj_input])
		# print "after agg", y.shape
		y = DynamicRoutingLayer(num_routing=1 if num_caps==1 else args.num_routing,
			name="dynamic_routing_layer_{}".format(i))(y)

		# y = layers.Lambda(squash, name="cap_layer_{}".format(i))(y)
		# y = GraphCapsuleLayer(num_capsule=num_caps, dim_capsule=capsule_dim, num_routing=num_routing, 
		# 	name="cap_layer_{}".format(i))(y)
		# y = AggGraphCapsuleLayer(num_neighbours=neighbourhood_sample_size+1,
		# 	num_capsule=num_caps, dim_capsule=capsule_dim, num_routing=num_routing, 
		# 	name="cap_layer_{}".format(i))(y)
		# print "y output", y.get_shape()
		feature_prob = Length(name="feature_prob_layer_{}".format(i))(y)
		# feature_prob = layers.ActivityRegularization(l2=0.0005,
		# 	name="activity_regularization_layer_{}".format(i))(feature_prob)
		# print "feat output", feature_prob.get_shape()
		# if num_caps == num_classes:
			# label_predictions.append(feature_prob)

		layer_embedding = Mask(name="embedding_mask_layer_{}".format(i))(y)
		layer_embedding = layers.Dense(2, activation=squash, name="embedding_dense_layer_{}".format(i))(layer_embedding)

		# layer_embedding = feature_prob
		# layer_embedding = layers.BatchNormalization(#gamma_initializer=Constant(5), 
		# 	beta_regularizer=l2(0.001),
		# 	name="embedding_normalization_layer_{}".format(i))(layer_embedding)
		# layer_embedding = layers.Lambda(embedding_function, 
		# 	output_shape=lambda x : x, name="embedding_layer_{}".format(i))(layer_embedding)

		# layer_embedding = layers.Reshape([num_caps*capsule_dim,], name="embedding_layer_{}".format(i))(y)
		# layer_embedding = layers.Lambda(squash, name="embedding_squash_layer_{}".format(i))(layer_embedding)


		embeddings.append(layer_embedding)
		# hyperbolic_distances.append(layer_hyperbolic_distance)

		# y = layers.Lambda(squash, name="squash_layer_{}".format(i))(y)

		if num_caps == num_classes:
			# feature_prob = Length(name="feature_prob_layer_{}".format(i))(y)
			label_predictions.append(feature_prob)

	masked_y = Mask(name="mask_layer")(y)
	reconstruction_hidden = layers.Dense(128, activation="relu",
		name="reconstruction_hidden")(masked_y)
	reconstruction_output = layers.Dense(data_dim,
		name="reconstruction_output")(reconstruction_hidden)

	losses = [masked_margin_loss]*len(label_predictions) +\
	[hyperbolic_negative_sampling_loss]*len(embeddings) + ["mse"]
	
	loss_weights = []
	if args.use_labels:
		loss_weights += [1.] * len(label_predictions)
	else:
		loss_weights += [0.] * len(label_predictions)
	if args.no_embedding_loss:
		loss_weights += [0.] * len(embeddings)
	elif args.no_intermediary_loss:
		loss_weights += [0.] * (len(embeddings) - 1)  + [1e-0]
	else:
		# loss_weights += [1e-0/len(hyperbolic_distances)]*len(hyperbolic_distances)
		loss_weights += [1e-0]*len(embeddings)
	loss_weights += [args.reconstruction_loss]

	print ("generating model with loss weights:", loss_weights)
	# raise SystemExit

	graphcaps = Model([x, adj_input], 
		label_predictions + embeddings + [reconstruction_output])
	adam = Adam( )
	graphcaps.compile(optimizer=adam, loss=losses, loss_weights=loss_weights)
	return graphcaps
