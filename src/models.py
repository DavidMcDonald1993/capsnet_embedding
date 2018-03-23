
import os
import numpy as np

# import tensorflow as tf
from keras import layers
from keras.models import Model, load_model
# from keras.regularizers import l2
from keras.optimizers import Adam
# from keras import backend as K

from graphcaps_layers import AggregateLayer, GraphCapsuleLayer, HyperbolicDistanceLayer, Length, squash
from losses import masked_crossentropy, masked_margin_loss, hyperbolic_negative_sampling_loss

def connect_layers(layer_tuples, x):
	
	y = x

	for layer_tuple in layer_tuples:
		for layer in layer_tuple:
			y = layer(y)

	return y

def load_models(X, Y, model_path, args):
	# neighbourhood_sample_sizes, num_primary_caps_per_layer, 
	# num_filters_per_layer, agg_dim_per_layer,
	# num_capsules_per_layer, capsule_dim_per_layer, args):



	# def get_model_memory_usage(batch_size, model):

	#     shapes_mem_count = 0
	#     for l in model.layers:
	#         single_layer_mem = 1
	#         for s in l.output_shape:
	#             if s is None:
	#                 continue
	#             single_layer_mem *= s
	#         if "cap_layer" in l.name:
	#         	_, n, in_cap, in_cap_dim = l.input_shape
	#         	_, _, out_cap, out_cap_dim = l.output_shape

	#         	in_hat_shape = [n, in_cap, out_cap, out_cap_dim]
	#         	c_shape = [n, out_cap, in_cap]
 #        		single_layer_mem += np.prod(in_hat_shape)
 #        		single_layer_mem *= np.prod(c_shape)

	#         shapes_mem_count += single_layer_mem

	#     trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
	#     non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

	#     total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
	#     gbytes = np.round(total_memory / (1024.0 ** 3), 3)
	#     return gbytes

	saved_models = sorted(os.listdir(model_path))
	initial_epoch = len(saved_models)

	data_dim = X.shape[1]
	num_classes = Y.shape[1]

	if initial_epoch == 0:

		print ("Creating new model")

		# batch_size = args.batch_size
		# num_positive_samples = 1
		# num_negative_samples = args.num_neg


		model = generate_graphcaps_model(data_dim, num_classes, args)
		# neighbourhood_sample_sizes, num_primary_caps_per_layer, num_filters_per_layer, agg_dim_per_layer,
		# num_capsules_per_layer, capsule_dim_per_layer, args)


	else:

		print ("Loading model from file")

		# batch_size = args.batch_size
		# num_positive_samples = 1
		# num_negative_samples = args.num_neg


		# model = generate_graphcaps_model(X, Y, batch_size, 
		# num_positive_samples, num_negative_samples,
		# neighbourhood_sample_sizes, num_primary_caps_per_layer, num_filters_per_layer, agg_dim_per_layer,
		# num_capsules_per_layer, capsule_dim_per_layer)

		# model.load_weights(os.path.join(model_path, saved_models[-1]))

		model = load_model(os.path.join(model_path, saved_models[-1]),
			custom_objects={"AggregateLayer":AggregateLayer, 
							"Length":Length,
							 "HyperbolicDistanceLayer":HyperbolicDistanceLayer, 
							 "GraphCapsuleLayer": GraphCapsuleLayer,
							 "hyperbolic_negative_sampling_loss": hyperbolic_negative_sampling_loss, 
							 "masked_margin_loss": masked_margin_loss})
		# for l in model.layers:
		# 	print l.name
		# 	print l.get_weights()
		# 	print
		# raise SystemExit

	# adam = Adam(lr=1e-5, clipnorm=1.)
	# model.compile(optimizer=adam, 
	# 	loss=[masked_crossentropy]*1 + 
	# 	[hyperbolic_negative_sampling_loss]*2, 
	# 	loss_weights=[0.]*1 + 
	# 	# [0]*(len(capsnet_distance_outputs)-1)+[1])
	# 	[1./2]*2)

	model.summary()
	# raise SystemExit

	# model_memory_usage = get_model_memory_usage(args.batch_size, model)
	# print "Memory usage: {}Gb".format(model_memory_usage)
	# assert model_memory_usage < 2, "This model will use {}Gb of memory. Consider decreasing the batch_size".format(model_memory_usage)
	# raise SystemExit

	embedder, label_prediction_model = build_embedder_and_prediction_model(data_dim, num_classes, model, args)

	return model, embedder, label_prediction_model, initial_epoch

def build_embedder_and_prediction_model(data_dim, num_classes, model, args):

	neighbourhood_sample_sizes = args.neighbourhood_sample_sizes
	number_of_capsules_per_layer = args.number_of_capsules_per_layer

	def model_to_dict(model):
		layer_dict = {}
		for layer in model.layers:
			layer_name = layer.name.split("_layer")[0]
			layer_dict.setdefault(layer_name, []).append(layer)
		return layer_dict

	layer_dict = model_to_dict(model)
	num_layers = len(neighbourhood_sample_sizes)

	embedding_layer = num_layers - 1
	embedder_lambda = lambda x, l=embedding_layer:\
	connect_layers(list(zip(layer_dict["agg"][:l], layer_dict["batch_normalization"][:l],\
	layer_dict["cap_input"][:l], layer_dict["cap"][:l], layer_dict["squash"][:l])) +\
	[(layer_dict["agg"][l], layer_dict["batch_normalization"][l], layer_dict["cap_input"][l], layer_dict["cap"][l]),\
	(layer_dict["embedding_reshape"][l], layer_dict["embedding_squash"][l])], x)

	embedder_input_num_neighbours = np.prod(neighbourhood_sample_sizes + 1)
	embedder_input = layers.Input(shape=(embedder_input_num_neighbours, 1, data_dim), name="embedder_input")
	embedder_output = embedder_lambda(embedder_input)

	embedder = Model(embedder_input, embedder_output)

	# embedder.summary()
	# raise SystemExit

	label_prediction_layers = np.where(number_of_capsules_per_layer==num_classes)[0] + 1
	if num_classes > 1 and len(label_prediction_layers) > 0:
		label_prediction_layer = label_prediction_layers[-1]
		label_prediction_input_num_neighbours = np.prod(neighbourhood_sample_sizes[:label_prediction_layer] + 1)
		label_prediction_input = layers.Input(shape=(label_prediction_input_num_neighbours, 1, data_dim),
			name="label_prediction_input")
		label_prediction_lambda = lambda x, i=len(label_prediction_layers)-1, l=label_prediction_layer: connect_layers(list(zip(layer_dict["agg"][:l], 
			layer_dict["batch_normalization"][:l], layer_dict["cap_input"][:l], layer_dict["cap"][:l], layer_dict["squash"][:l])) + 
			[(layer_dict["label_prediction"][i], )], x)
		label_prediction_output = label_prediction_lambda(label_prediction_input)

		label_prediction_model = Model(label_prediction_input, label_prediction_output)
	else: 
		label_prediction_model = None

	return embedder, label_prediction_model

def generate_graphcaps_model(data_dim, num_classes, args):
	# neighbourhood_sample_sizes, num_primary_caps_per_layer, num_filters_per_layer, agg_dim_per_layer,
	# number_of_capsules_per_layer, capsule_dim_per_layer, args):

	neighbourhood_sample_sizes = args.neighbourhood_sample_sizes
	num_primary_caps_per_layer = args.num_primary_caps_per_layer
	num_filters_per_layer = args.num_filters_per_layer
	agg_dim_per_layer = args.agg_dim_per_layer
	number_of_capsules_per_layer = args.number_of_capsules_per_layer
	capsule_dim_per_layer = args.capsule_dim_per_layer

	num_positive_samples = args.num_positive_samples
	num_negative_samples = args.num_negative_samples


	# N, data_dim = X.shape
	# _, num_classes = Y.shape

	num_layers = len(neighbourhood_sample_sizes)
	output_size = 1 + num_positive_samples + num_negative_samples
	num_neighbours_per_layer = np.array([np.prod(neighbourhood_sample_sizes[i:]+1) * output_size for i in range(num_layers+1)])
	number_of_capsules_per_layer = np.append(1, number_of_capsules_per_layer)


	x = layers.Input(shape=(num_neighbours_per_layer[0], 1, data_dim), 
		name="input_layer")
	y = x

	embeddings = []
	hyperbolic_distances = []
	label_predictions = []



	for i, neighbourhood_sample_size, num_primary_caps, num_filters, agg_dim, num_caps, capsule_dim in zip(range(num_layers),
		neighbourhood_sample_sizes, num_primary_caps_per_layer, num_filters_per_layer, agg_dim_per_layer, 
		number_of_capsules_per_layer[1:], capsule_dim_per_layer):

		y = AggregateLayer(num_neighbours=neighbourhood_sample_size+1, num_caps=num_primary_caps,
			num_filters=num_filters, new_dim=agg_dim,
			activation="relu", name="agg_layer_{}".format(i))(y)
		y = layers.BatchNormalization(axis=2, name="batch_normalization_layer_{}".format(i))(y)
		
		# y = layers.Reshape(tf.stack([tf.shape(y)[0], tf.shape(y)[1], number_of_capsules_per_layer[i], num_filters*agg_dim]),
		# 	name="cap_input_layer")(y)
		# y = layers.Lambda(lambda tensor, shape=[num_primary_caps, num_filters*agg_dim]: 
		# 	K.map_fn(lambda x1, shape=shape: 
		# 	K.map_fn(lambda x2, shape=shape: K.reshape(x2, shape=shape), elems=x1), elems=tensor),
		# 	name="cap_input_layer_{}".format(i))(y)
			# K.reshape(tensor, shape=tf.stack([K.shape(tensor)[0], K.shape(tensor)[1]] + 
			# 	shape)))(y)
		y = layers.Reshape([-1, num_primary_caps, num_filters*agg_dim], name="cap_input_layer_{}".format(i))(y)
		
		if num_caps == 1:
			num_routing = 1
		else:
			num_routing = 3 

		y = GraphCapsuleLayer(num_capsule=num_caps, dim_capsule=capsule_dim, num_routing=num_routing, 
			name="cap_layer_{}".format(i))(y)

		layer_embedding = layers.Reshape([-1, num_caps*capsule_dim], name="embedding_reshape_layer_{}".format(i))(y)
		layer_embedding = layers.Lambda(squash, name="embedding_squash_layer_{}".format(i))(layer_embedding)
		layer_hyperbolic_distance = HyperbolicDistanceLayer(num_positive_samples=num_positive_samples,
			num_negative_samples=num_negative_samples, name="hyperbolic_distance_layer_{}".format(i))(layer_embedding)

		embeddings.append(layer_embedding)
		hyperbolic_distances.append(layer_hyperbolic_distance)

		y = layers.Lambda(squash, name="squash_layer_{}".format(i))(y)

		if num_caps == num_classes:
			label_prediction = Length(name="label_prediction_layer_{}".format(i))(y)
			label_predictions.append(label_prediction)

		# agg_layers.append(AggregateLayer(num_neighbours=neighbourhood_sample_size+1, num_filters=num_filters, new_dim=agg_dim,
		# 	activation="relu", name="agg_layer_{}".format(i)))
		# # normalizer over filter/channel dimension
		# normalization_layers.append(layers.BatchNormalization(axis=2, name="batch_normalization_layer_{}".format(i)))
		# capsule_layers.append(GraphCapsuleLayer(num_capsule=num_caps, dim_capsule=capsule_dim, num_routing=num_routing, 
		# 	name="cap_layer_{}".format(i)))

		# capsule_outputs.append(layers.Lambda(squash, name="squash_layer_{}".format(i)))

		# if num_caps == num_classes:
		# 	label_predictions.append(Length(name="label_prediction_layer_{}".format(i)))

		# reshape_layers.append(layers.Reshape((-1, num_caps*capsule_dim), name="embedding_reshape_layer_{}".format(i)))
		# embeddings.append(layers.Lambda(squash, name="embedding_squash_layer_{}".format(i)))
		# hyperbolic_distances.append(HyperbolicDistanceLayer(num_positive_samples=num_positive_samples,
		# 	num_negative_samples=num_negative_samples, name="hyperbolic_distance_layer_{}".format(i)))


	# label_prediction_layers = np.where(number_of_capsules_per_layer==num_classes)[0] + 1

	## lambda functions of input layer
	# label_prediction_lambdas = [lambda x, i=i, l=l: 
	# connect_layers(zip(agg_layers[:l], 
	# 	normalization_layers[:l], 
	# 	capsule_layers[:l], 
	# 	capsule_outputs[:l]) +
	# 	[(label_predictions[i], )], x) for i, l in enumerate(label_prediction_layers)]


	# embedder_lambdas = [lambda x, l=l: connect_layers(zip(agg_layers[:l], 
	# 	normalization_layers[:l], 
	# 	capsule_layers[:l], 
	# 	capsule_outputs[:l]) + 
	# 	[(agg_layers[l], 
	# 		normalization_layers[l], 
	# 		capsule_layers[l] 
	# 		),
	# 	(reshape_layers[l], embeddings[l])], x) for l in range(num_layers)]

	# capsnet_label_prediction_outputs = [label_prediction_lambda(training_input) for 
	# 	label_prediction_lambda in label_prediction_lambdas]

	# capsnet_embedding_outputs = [embedder_lambda(training_input) for 
	# 	embedder_lambda in embedder_lambdas]
	# capsnet_distance_outputs = [hyperbolic_distance(embedding) for 
	# 	hyperbolic_distance, embedding in zip(hyperbolic_distances, capsnet_embedding_outputs)]

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
		loss_weights += [0.] * (len(hyperbolic_distances) - 1)  + [1e-2]
	else:
		loss_weights += [1e-2/len(hyperbolic_distances)]*len(hyperbolic_distances)

	graphcaps = Model(x,  label_predictions + hyperbolic_distances)
	adam = Adam(lr=1e-4, clipnorm=1.)
	graphcaps.compile(optimizer=adam, loss=losses, loss_weights=loss_weights)

	# graphcaps.summary()
	# raise SystemExit

	return graphcaps



	# embedder_input = layers.Input(shape=(np.prod(neighbourhood_sample_sizes + 1), 1, data_dim), name="embedder_input")
	# embedder_output = embedder_lambdas[-1](embedder_input)

	# embedder = Model(embedder_input, embedder_output)

	# if len(label_prediction_layers) > 0:
	# 	label_prediction_input = layers.Input(shape=(np.prod(neighbourhood_sample_sizes[:label_prediction_layers[-1]] + 1), 1, data_dim),
	# 		name="label_prediction_input")
	# 	label_prediction_output = label_prediction_lambdas[-1](label_prediction_input)

	# 	prediction_model = Model(label_prediction_input, label_prediction_output)
	# else: 
	# 	prediction_model = None

	# return graphcaps, embedder, prediction_model

# def build_graphcaps(data_dim, num_classes, embedding_dim, num_positive_samples, num_negative_samples, sample_sizes):

# 	output_size = 1 + num_positive_samples + num_negative_samples
	
# 	num_capsules_first_layer = 16
# 	capsule_dim_first_layer = 8

# 	num_capsules_second_layer = num_classes
# 	capsule_dim_second_layer = 4

# 	num_capsules_third_layer = 1
# 	capsule_dim_third_layer = embedding_dim

# 	x = layers.Input(shape=(np.prod(sample_sizes + 1) * output_size, 1, data_dim), name="input_signal")
# 	# x_norm = layers.BatchNormalization()(x)
# 	agg1 = AggregateLayer(num_neighbours=sample_sizes[-1] + 1, num_filters=16, new_dim=128,
# 								activation="relu", name="agg1")(x)
# 	cap1 = GraphCapsuleLayer(num_capsule=num_capsules_first_layer, dim_capsule=capsule_dim_first_layer, 
# 		num_routing=3, name="cap1")(agg1)



# 	cap1_squash = layers.Lambda(squash, output_shape=lambda x: x, name="cap1_squash")(cap1)
# 	# label_predictions = Length(name="label_predictions")(cap1_squash)



# 	# cap1_length = Length(name="length_first_layer")(cap1)
# 	embedding_first_layer = layers.Reshape((-1, num_capsules_first_layer*capsule_dim_first_layer),
# 		name="reshape_first_layer")(cap1)
# 	embedding_first_layer = layers.Lambda(squash, output_shape=lambda x : x, name="embedding_first_layer")(embedding_first_layer)
# 	hyperbolic_distance_first_layer = HyperbolicDistanceLayer(num_positive_samples=num_positive_samples,
# 		num_negative_samples=num_negative_samples, name="hyperbolic_distance_first_layer")(embedding_first_layer)







# 	agg2 = AggregateLayer(num_neighbours=sample_sizes[-2] + 1, num_filters=16, new_dim=32,
# 								activation="relu", name="agg2")(cap1_squash)
# 	cap2 = GraphCapsuleLayer(num_capsule=num_capsules_second_layer, dim_capsule=capsule_dim_second_layer, 
# 		num_routing=3, name="cap2")(agg2)

# 	# cap2_length = Length(name="cap2_length")(cap2)
# 	cap2_squash = layers.Lambda(squash, output_shape=lambda x: x)(cap2)
# 	label_predictions = Length(name="label_predictions")(cap2_squash)

# 	embedding_second_layer = layers.Reshape((-1, num_capsules_second_layer*capsule_dim_second_layer),
# 		name="reshape_second_layer")(cap2)
# 	embedding_second_layer = layers.Lambda(squash, output_shape=lambda x : x, name="embedding_second_layer")(embedding_second_layer)
# 	hyperbolic_distance_second_layer = HyperbolicDistanceLayer(num_positive_samples=num_positive_samples,
# 		num_negative_samples=num_negative_samples, name="hyperbolic_distance_second_layer")(embedding_second_layer)




# 	agg3 = AggregateLayer(num_neighbours=sample_sizes[-3] + 1, num_filters=8, new_dim=16,
# 								activation="relu", name="agg3")(cap2_squash)
# 	cap3 = GraphCapsuleLayer(num_capsule=num_capsules_third_layer, dim_capsule=capsule_dim_third_layer, 
# 		num_routing=1, name="cap3")(agg3)

# 	# cap3_length = Length(name="cap3_length")(cap3)
# 	embedding_third_layer = layers.Reshape((-1, num_capsules_third_layer*capsule_dim_third_layer),
# 		name="reshape_third_layer")(cap3)
# 	embedding_third_layer = layers.Lambda(squash, output_shape=lambda x : x, name="embedding_third_layer")(embedding_third_layer)
# 	hyperbolic_distance_third_layer = HyperbolicDistanceLayer(num_positive_samples=num_positive_samples,
# 		num_negative_samples=num_negative_samples, name="hyperbolic_distance_third_layer")(embedding_third_layer)

	

# 	model = Model(x, #[hyperbolic_distance_first_layer])#, hyperbolic_distance_second_layer])
# 		[label_predictions, label_predictions, 
# 		hyperbolic_distance_first_layer, hyperbolic_distance_second_layer, hyperbolic_distance_third_layer]) 
# 	model.compile(optimizer="adam", 
# 		# loss=[hyperbolic_negative_sampling_loss]*1)
# 		loss=[masked_margin_loss] * 2 + [hyperbolic_negative_sampling_loss] * 3,
# 		# hyperbolic_negative_sampling_loss, hyperbolic_negative_sampling_loss], 
# 		loss_weights=[0.0, 0.0, 0.333, 0.33333, 0.333333])

# 	embedder = Model(x, embedding_third_layer)

# 	return model, embedder