import random
import numpy as np
import scipy as sp
# import networkx as nx

# from itertools import izip_longest

# from utils import create_neighbourhood_sample_list
from data_utils import preprocess_data
# from node2vec_sampling import Graph


def get_neighbourhood_samples(nodes, neighbourhood_sample_sizes, neighbours):

	neighbourhood_sample_list = [nodes]

	for neighbourhood_sample_size in neighbourhood_sample_sizes[::-1]:

		neighbourhood_sample_list.append(np.array([np.concatenate([np.append(n, 
			np.random.choice(np.append(n, neighbours[n]), 
			replace=True, size=neighbourhood_sample_size)) for n in batch]) for batch in neighbourhood_sample_list[-1]]))

	# flip neighbour list
	neighbourhood_sample_list = neighbourhood_sample_list[::-1]

	# only return input_nodes
	# input_nodes = neighbourhood_sample_list[0]

	return neighbourhood_sample_list

def neighbourhood_sample_generator(G, X, Y, train_mask, 
	positive_samples, ground_truth_negative_samples, args):
	# neighbourhood_sample_sizes, num_capsules_per_layer,
	# num_positive_samples, num_negative_samples, batch_size,):
	
	'''
	performs node2vec style neighbourhood sampling for positive samples.
	negative samples are selected according to degree
	uniform sampling of neighbours for aggregation

	'''

	number_of_capsules_per_layer = args.number_of_capsules_per_layer
	neighbourhood_sample_sizes = args.neighbourhood_sample_sizes
	batch_size = args.batch_size

	num_positive_samples = args.num_positive_samples
	num_negative_samples = args.num_negative_samples

	num_classes = Y.shape[1]
	label_prediction_layers = np.where(number_of_capsules_per_layer==num_classes)[0] + 1

	neighbours = {n : list(G.neighbors(n)) for n in G.nodes()}

	num_embeddings = neighbourhood_sample_sizes.shape[0]

	num_steps = int((len(positive_samples) + batch_size - 1) // batch_size)
	
	while True:

		random.shuffle(positive_samples)

		for step in range(num_steps):

			batch_positive_samples = np.array(positive_samples[step * batch_size : (step + 1) * batch_size])
			batch_negative_samples =\
				np.array([np.random.choice(ground_truth_negative_samples[u], replace=True, size=(num_negative_samples,))\
					for u in batch_positive_samples[:,0]])
			batch_nodes = np.append(batch_positive_samples, batch_negative_samples, axis=1)
			# batch_nodes = batch_sampler.next()
			# batch_nodes = np.array(batch_nodes)
			# batch_nodes = np.random.permutation(batch_nodes)

			neighbourhood_sample_list = get_neighbourhood_samples(batch_nodes, neighbourhood_sample_sizes, neighbours)

			# shape is [batch_size, output_shape*prod(sample_sizes), D]
			input_nodes = neighbourhood_sample_list[0]
			original_shape = list(input_nodes.shape)

			if sp.sparse.issparse(X):

				x = X[input_nodes.flatten()].toarray()
				x = preprocess_data(x)

			else:
				x = X[input_nodes]
			# add artificial capsule dimension 
			x = x.reshape(original_shape + [1, -1])
			# x = np.expand_dims(x, 2)
			# shape is now [batch_nodes, output_shape*prod(sample_sizes), 1, D]

			masked_labels = []
			# all_zero_mask = False
			for layer in label_prediction_layers:
				nodes_to_evaluate_label = neighbourhood_sample_list[layer]
				original_shape = list(nodes_to_evaluate_label.shape)
				y = Y[nodes_to_evaluate_label.flatten()]#.toarray()
				y = y.reshape(original_shape + [-1])
				# print y.shape, sp.sparse.issparse(y)
				# raise SystemExit


				mask = train_mask[neighbourhood_sample_list[layer]]
				# if (mask == 0).all():
					# all_zero_mask = True
					# break
				y_masked = np.append(mask, y, axis=-1)
				# print y_masked.shape
				# raise SystemExit
				masked_labels.append(y_masked)

			negative_sample_targets = np.zeros((batch_nodes.shape[0], num_positive_samples+num_negative_samples))
			negative_sample_targets[:,0] = 1
			negative_sample_targets = [negative_sample_targets] * num_embeddings


			# if not all_zero_mask:
			yield x, masked_labels + negative_sample_targets


def validation_generator(validation_callback, G, X, idx, neighbourhood_sample_sizes, num_steps, batch_size=100):
	neighbours = {n: list(G.neighbors(n)) for n in G.nodes()}
	while True:
		# np.random.shuffle(nodes_to_val)
		random.shuffle(idx)
		nodes_to_val = np.array(idx).reshape(-1, 1)
		for step in range(num_steps):			
			batch_nodes = nodes_to_val[batch_size*step : batch_size*(step+1)]
			neighbourhood_sample_list = get_neighbourhood_samples(batch_nodes, neighbourhood_sample_sizes, neighbours)
			input_nodes = neighbourhood_sample_list[0]
			if sp.sparse.issparse(X):
				x = X[input_nodes.flatten()].toarray()
				x = preprocess_data(x)
			else:
				x = X[input_nodes]
			yield x.reshape([-1, input_nodes.shape[1], 1, X.shape[-1]])
			if step == 0:
				# save order of nodes for evaluation 
				validation_callback.nodes_to_val = idx[:]


# def prediction_generator(G, X, nodes_to_predict, neighbourhood_sample_sizes, num_steps, batch_size=100):
# 	neighbours = {n: list(G.neighbors(n)) for n in G.nodes()}
# 	while True:
# 		np.random.shuffle(nodes_to_predict)
# 		for step in range(num_steps):
# 			batch_nodes = nodes_to_predict[batch_size*step : batch_size*(step+1)]
# 			neighbourhood_sample_list = get_neighbourhood_samples(batch_nodes, neighbourhood_sample_sizes, neighbours)
# 			input_nodes neighbourhood_sample_list[0]
# 			if sp.sparse.issparse(X):
# 				x = X[batch_nodes.flatten()].toarray()
# 				x = preprocess_data(x)
# 			else:
# 				x = X[batch_nodes]
# 			yield x.reshape([-1, input_nodes.shape[1], 1, X.shape[-1]])
