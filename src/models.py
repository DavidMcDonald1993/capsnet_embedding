
import os
import re
import numpy as np

import tensorflow as tf
from keras import layers
from keras.models import Model, load_model
from keras.regularizers import l1, l2, Regularizer
from keras.optimizers import Adam
from keras.initializers import RandomUniform, Constant
import keras.backend as K

from graphcaps_layers import UHatLayer, NeighbourhoodSamplingLayer, DynamicRoutingLayer
from graphcaps_layers import Mask, ProbabilisticMask, EmbeddingCapsuleLayer
from graphcaps_layers import Length, squash, embedding_function
# from graphcaps_layers import SimpleAggregateLayer, AggregateLayer, AggGraphCapsuleLayer, GraphCapsuleLayer, HyperbolicDistanceLayer, Length, squash, embedding_function
from losses import masked_margin_loss, unsupervised_margin_loss, hyperbolic_negative_sampling_loss, probabilistic_negative_sampling_loss
# from regularizers import SparseActivityRegularizer

def generate_graphcaps_model(data_dim, num_classes, args):

	neighbourhood_sample_sizes = args.neighbourhood_sample_sizes
	number_of_capsules_per_layer = args.number_of_capsules_per_layer
	capsule_dim_per_layer = args.capsule_dim_per_layer

	embedding_dims = args.embedding_dims

	num_positive_samples = args.num_positive_samples
	num_negative_samples = args.num_negative_samples

	num_layers = len(neighbourhood_sample_sizes)
	output_size = 1 + num_positive_samples + num_negative_samples
	num_neighbours_per_layer = np.array([np.prod(neighbourhood_sample_sizes[i:]+1) * output_size for i in range(num_layers+1)])


	x = layers.Input(shape=(data_dim, ), name="feats_input")
	y = x

	adj_input = layers.Input(shape=(None,), name="adj_input")

	if args.num_primary_caps is not None:
		y = layers.Dense(args.num_primary_caps * args.primary_cap_dim, activation=None, 
			# kernel_initializer=RandomUniform(-0.1, 0.1),
			kernel_regularizer=l2(1e-30), name="primary_cap_layer")(y)
		y = layers.Reshape([args.num_primary_caps, args.primary_cap_dim], name="primary_reshape_layer")(y)
		y = layers.Lambda(squash, output_shape=lambda x: x, 
			name="primary_squash_layer")(y)
	else:
		raise Exception

	feature_probs = []
	embeddings = []

	for i, neighbourhood_sample_size, num_caps, capsule_dim, embedding_dim in zip(range(num_layers),
		neighbourhood_sample_sizes, number_of_capsules_per_layer, capsule_dim_per_layer, embedding_dims):

		y = UHatLayer(num_capsule=num_caps, dim_capsule=capsule_dim, use_bias=False,
			name="u_hat_layer_{}".format(i))(y)
		y = NeighbourhoodSamplingLayer(sample_size=neighbourhood_sample_size,
			name="neighbourhood_sample_layer_{}".format(i))([y, adj_input])
		y = DynamicRoutingLayer(num_routing=1 if num_caps==1 else args.num_routing,
			name="dynamic_routing_layer_{}".format(i))(y)

		feature_prob = Length(name="feature_prob_layer_{}".format(i),)(y)
		# mask = ProbabilisticMask(name="feature_mask_layer_{}".format(i))(feature_prob)

		# feature_prob = layers.Concatenate(name="concat_layer_{}".format(i))([mask, feature_prob])

		# feature_prob = layers.ActivityRegularization(l1=1e-3,
		# 	name="activity_regularization_layer_{}".format(i))(feature_prob)
		
		layer_embedding = EmbeddingCapsuleLayer(embedding_dim=embedding_dim, use_bias=True,
			name="capsule_embedding_layer_{}".format(i))(y)
		# layer_embedding = Mask(name="embedding_mask_layer_{}".format(i))(y)

		# layer_embedding = layers.Dense(embedding_dim, activation=squash, 
			# name="embedding_dense_layer_{}".format(i))(layer_embedding)
		# print layer_embedding.shape
		# raise SystemExit
		feature_probs.append(feature_prob)
		embeddings.append(layer_embedding)


	masked_y = Mask(name="mask_layer")(y)
	reconstruction_hidden = layers.Dense(args.reconstruction_dim, activation="relu",
		name="reconstruction_hidden")(masked_y)
	reconstruction_output = layers.Dense(data_dim,
		name="reconstruction_output")(reconstruction_hidden)

	model_inputs = [x, adj_input]
	model_outputs = feature_probs + embeddings + [reconstruction_output]

	losses = []
	# feature prob outputs
	losses += [
		masked_margin_loss 
		if args.use_labels and num_caps == num_classes 
		# else unsupervised_margin_loss 
		else probabilistic_negative_sampling_loss
		for num_caps in number_of_capsules_per_layer
	]
	# embedding losses
	losses += [hyperbolic_negative_sampling_loss] * len(embeddings)
	# reconstruction loss
	losses += ["mse"]
	
	loss_weights = []
	# feature prob loss weights
	loss_weights += [
		args.feature_prob_loss_weight
		if not args.use_labels or (args.use_labels and num_caps == num_classes)
		else 0
		for num_caps in number_of_capsules_per_layer
	]
	# embedding loss weights
	loss_weights += [
		args.embedding_loss_weight
		if not args.no_embedding_loss
		else 0
		for _ in number_of_capsules_per_layer

	]
	# reconstruction
	loss_weights += [args.reconstruction_loss_weight if not args.no_reconstruction else 0]
	# if args.use_labels:
	# 	loss_weights += [1.] * len(label_predictions)
	# else:
	# 	loss_weights += [0.] * len(label_predictions)
	# if args.no_embedding_loss:
	# 	loss_weights += [0.] * len(embeddings)
	# elif args.no_intermediary_loss:
	# 	loss_weights += [0.] * (len(embeddings) - 1)  + [1e-0]
	# else:
	# 	# loss_weights += [1e-0/len(hyperbolic_distances)]*len(hyperbolic_distances)
	# 	loss_weights += [1e-0]*len(embeddings)

	print ("generating model with loss weights:", loss_weights)

	graphcaps = Model(model_inputs, model_outputs)
	# graphcaps.summary()
	# raise SystemExit
	# adam = Adam()
	graphcaps.compile(optimizer="adam", loss=losses, loss_weights=loss_weights)
	return graphcaps

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
							"EmbeddingCapsuleLayer": EmbeddingCapsuleLayer,
							"hyperbolic_negative_sampling_loss": hyperbolic_negative_sampling_loss, 
							"masked_margin_loss": masked_margin_loss,
							"unsupervised_margin_loss": unsupervised_margin_loss,
							"probabilistic_negative_sampling_loss": probabilistic_negative_sampling_loss,
							"tf":tf,
							"min_norm": 1e-7,
							"max_norm":np.nextafter(1, 0, dtype=K.floatx()),
							"max_": np.finfo(K.floatx()).max})
		# K.set_value(model.optimizer.lr, 1e-5)
		# print K.get_value(model.optimizer.lr)
		# raise SystemExit


	model.summary()
	# raise SystemExit

	return model, initial_epoch
	