import random
import numpy as np
import scipy as sp
# import networkx as nx

# from itertools import izip_longest

# from utils import create_neighbourhood_sample_list
from data_utils import preprocess_data
# from node2vec_sampling import Graph


def get_neighbourhood_samples(nodes, neighbourhood_sample_sizes, neighbours):

	'''
	generates a list of a sample of the neighbours of each node in the batch
	'''

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

		skip = 0
		for step in range(num_steps):

			batch_positive_samples = np.array(positive_samples[step * batch_size : (step + 1) * batch_size])
			batch_negative_samples =\
				np.array([np.random.choice(ground_truth_negative_samples[u], replace=True, size=(num_negative_samples,))\
					for u in batch_positive_samples[:,0]])
			batch_nodes = np.append(batch_positive_samples, batch_negative_samples, axis=1)

			# print batch_nodes
			# for u, v, neg in zip(batch_nodes[:,0], batch_nodes[:,1], batch_nodes[:,2:]):
			# 	# print u, v, neg
			# 	assert (u, v) in positive_samples
			# 	assert all([n in ground_truth_negative_samples[u] for n in neg])
			# print batch_nodes
			# for u in batch_nodes[:,0]:
			# 	print u, ground_truth_negative_samples[u]

			neighbourhood_sample_list = get_neighbourhood_samples(batch_nodes, neighbourhood_sample_sizes, neighbours)
			# print neighbourhood_sample_sizes
			# for l in neighbourhood_sample_list:
			# 	print l.shape
			# for size, ns1, ns2 in zip(neighbourhood_sample_sizes, neighbourhood_sample_list, neighbourhood_sample_list[1:]):
			# 	for i, _ in enumerate(ns1):
			# 		for j, v in enumerate(ns1[i]):
			# 			neigh = ns2[i, j//(size+1)]
			# 			assert v == neigh or neigh in neighbours[v], "incorrect neighbourhood samples" 

			# shape is [batch_size, output_shape*prod(sample_sizes), D]
			input_nodes = neighbourhood_sample_list[0]
			original_shape = list(input_nodes.shape)

			if sp.sparse.issparse(X):

				x = X[input_nodes.flatten()].toarray()
				x = preprocess_data(x)

			else:
				x = X[input_nodes]
			# add artificial capsule dimension 
			x = x.reshape(original_shape + [-1])
			# print x.shape
			# assert np.allclose(x.argmax(axis=-1), input_nodes) 
			# print x.argmax(axis=-1)
			# print input_nodes
			# shape is now [batch_nodes, output_shape*prod(sample_sizes), 1, D]

			masked_labels = []
			all_zero_mask = False
			for layer in label_prediction_layers:
				nodes_to_evaluate_label = neighbourhood_sample_list[layer]
				original_shape = list(nodes_to_evaluate_label.shape)
				y = Y[nodes_to_evaluate_label.flatten()]#.toarray()
				# for y_prime, node in zip(y, nodes_to_evaluate_label.flatten()):
				# 	print node
				# 	print "y_prime", y_prime
				# 	print Y[node].toarray()
					# assert np.allclose(y_prime, Y[node].toarray().flatten())

				if sp.sparse.issparse(y):
					y = y.toarray()
				y = y.reshape(original_shape + [-1])

				mask = train_mask[nodes_to_evaluate_label]
				# print "mask", mask.shape
				assert mask.shape == tuple(list(nodes_to_evaluate_label.shape) + [1])
				# print mask
				all_zeros = not mask.any()
				if all_zeros:
					all_zero_mask = True
				y_masked = np.append(mask, y, axis=-1)
				assert y_masked.shape == tuple(list(nodes_to_evaluate_label.shape) + [1 + num_classes])
				# print "y_masked", y_masked.shape, mask.sum(), y_masked[:,:,0].sum()
				# print y_masked.reshape(-1, y_masked.shape[-1])
				masked_labels.append(y_masked)

			negative_sample_targets = np.zeros((batch_nodes.shape[0], num_positive_samples+num_negative_samples))
			negative_sample_targets[:,0] = 1
			negative_sample_targets = [negative_sample_targets] * num_embeddings


			if not all_zero_mask:
				yield x, masked_labels + negative_sample_targets
			else:
				skip +=1
				# print "skip", skip
		print "skipped {}/{}".format(skip, num_steps)
		# raise SystemExit



def validation_generator(validation_callback, G, X, idx, neighbourhood_sample_sizes, num_steps, batch_size=100):
	'''
	generator that yields input data for validation
	'''
	neighbours = {n: list(G.neighbors(n)) for n in G.nodes()}
	while True:
		# np.random.shuffle(nodes_to_val)
		# random.shuffle(idx)
		nodes_to_val = np.array(idx).reshape(-1, 1)
		for step in range(num_steps):			
			batch_nodes = nodes_to_val[batch_size*step : batch_size*(step+1)]
			assert batch_nodes.shape[1] == 1
			neighbourhood_sample_list = get_neighbourhood_samples(batch_nodes, neighbourhood_sample_sizes, neighbours)

			# for size, ns1, ns2 in zip(neighbourhood_sample_sizes, neighbourhood_sample_list, neighbourhood_sample_list[1:]):
			# 	for i, _ in enumerate(ns1):
			# 		for j, v in enumerate(ns1[i]):
			# 			neigh = ns2[i, j//(size+1)]
			# 			assert v == neigh or neigh in neighbours[v], "incorrect neighbourhood samples in val gen" 

			input_nodes = neighbourhood_sample_list[0]
			if sp.sparse.issparse(X):
				x = X[input_nodes.flatten()].toarray()
				x = preprocess_data(x)
			else:
				x = X[input_nodes]
			yield x.reshape([-1, input_nodes.shape[1], X.shape[-1]])
			print "yielding step {}/{}".format(step, num_steps)
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
