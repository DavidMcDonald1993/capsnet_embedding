import random
import numpy as np
import scipy as sp
# import networkx as nx

# from itertools import izip_longest

from utils import create_neighbourhood_sample_list
from data_utils import preprocess_data
# from node2vec_sampling import Graph

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

	num_steps = (len(positive_samples) + batch_size - 1) // batch_size
	
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

			neighbour_list = create_neighbourhood_sample_list(batch_nodes, neighbourhood_sample_sizes, neighbours)

			# shape is [batch_size, output_shape*prod(sample_sizes), D]
			input_nodes = neighbour_list[0]
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
				nodes_to_evaluate_label = neighbour_list[layer]
				original_shape = list(nodes_to_evaluate_label.shape)
				y = Y[nodes_to_evaluate_label.flatten()]#.toarray()
				y = y.reshape(original_shape + [-1])
				# print y.shape, sp.sparse.issparse(y)
				# raise SystemExit


				mask = train_mask[neighbour_list[layer]]
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

# def generate_samples_node2vec(G, walks, num_positive_samples, num_negative_samples, context_size,):
# 	# p, q, num_walks, walk_length):

# 	N = nx.number_of_nodes(G)
# 	frequencies = np.array(dict(G.degree()).values()) ** 0.75

# 	# nx.set_edge_attributes(G=G, name="weight", values=1)
# 	# node2vec_graph = Graph(nx_G=G, is_directed=False, p=p, q=q)
# 	# # node2vec_graph.preprocess_transition_probs()
# 	# walks = node2vec_graph.simulate_walks(num_walks=num_walks, walk_length=walk_length)

# 	while True:
		
# 		random.shuffle(walks)

# 		for walk in walks:

# 			possible_negative_samples = np.setdiff1d(np.arange(N), walk)
# 			possible_negative_sample_frequencies = frequencies[possible_negative_samples]
# 			possible_negative_sample_frequencies /= possible_negative_sample_frequencies.sum()
# 			# negative_samples = np.random.choice(possible_negative_samples, replace=True, size=num_negative_samples, 
# 			# 				p=possible_negative_sample_frequencies / possible_negative_sample_frequencies.sum())

# 			for i in range(len(walk)):
# 				for j in range(i+1, min(len(walk), i+1+context_size)):
# 					if walk[i] == walk[j]:
# 						continue
# 					pair = np.array([walk[i], walk[j]])
# 					for negative_samples in np.random.choice(possible_negative_samples, replace=True, 
# 						size=(1+num_positive_samples, num_negative_samples),
# 						p=possible_negative_sample_frequencies):
# 						yield np.append(pair, negative_samples)
# 						pair = pair[::-1]

# def grouper(n, iterable, fillvalue=None):
# 	'''grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx'''
# 	args = [iter(iterable)] * n
# 	return izip_longest(fillvalue=fillvalue, *args)