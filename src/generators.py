import numpy as np
import networkx as nx

from itertools import izip_longest

from utils import compute_label_mask, create_neighbourhood_sample_list
from node2vec_sampling import Graph

def neighbourhood_sample_generator(G, X, Y, neighbourhood_sample_sizes, num_capsules_per_layer,
	num_positive_samples, num_negative_samples, context_size, batch_size, p, q, num_walks, walk_length,
	num_samples_per_class=20):
	
	'''
	performs node2vec style neighbourhood sampling for positive samples.
	negative samples are selected according to degree
	uniform sampling of neighbours for aggregation

	'''
	'''
	PRECOMPUTATION
	
	'''

	G = nx.convert_node_labels_to_integers(G)

	num_classes = Y.shape[1]
	if num_samples_per_class is not None:
		label_mask = compute_label_mask(Y, num_patterns_to_keep=num_samples_per_class)
	else:
		label_mask = np.ones(Y.shape)

	label_prediction_layers = np.where(num_capsules_per_layer==num_classes)[0] + 1

	
	neighbours = {n : list(G.neighbors(n)) for n in G.nodes()}
	

	num_layers = neighbourhood_sample_sizes.shape[0]

	node2vec_sampler = generate_samples_node2vec(G, num_positive_samples, num_negative_samples, context_size, 
		p, q, num_walks, walk_length)
	batch_sampler = grouper(batch_size, node2vec_sampler)
	
	'''
	END OF PRECOMPUTATION
	'''
	
	while True:

		batch_nodes = batch_sampler.next()
		batch_nodes = np.array(batch_nodes)

		neighbour_list = create_neighbourhood_sample_list(batch_nodes, neighbourhood_sample_sizes, neighbours)

		# shape is [batch_size, output_shape*prod(sample_sizes), D]
		x = X[neighbour_list[0]]
		x = np.expand_dims(x, 2)
		# shape is now [batch_nodes, output_shape*prod(sample_sizes), 1, D]


		negative_sample_targets = [Y[nl].argmax(axis=-1) for nl in neighbour_list[1:]]

		labels = []
		for layer in label_prediction_layers:
			y = Y[neighbour_list[layer]]
			mask = label_mask[neighbour_list[layer]]
			y_masked = np.append(mask, y, axis=-1)
			labels.append(y_masked)


		if all([(y_masked[:,:,:num_classes] > 0).any() for y_masked in labels]):
			yield x, labels + negative_sample_targets

def generate_samples_node2vec(G, num_positive_samples, num_negative_samples, context_size,
	p, q, num_walks, walk_length):

	nx.set_node_attributes(G=G, name="weight", values=1)
	
	N = nx.number_of_nodes(G)

	frequencies = np.array(dict(G.degree()).values()) ** 0.75

	node2vec_graph = Graph(nx_G=G, is_directed=False, p=p, q=q)
	node2vec_graph.preprocess_transition_probs()

	while True:
		walks = node2vec_graph.simulate_walks(num_walks=num_walks, walk_length=walk_length)
		for walk in walks:

			possible_negative_samples = np.setdiff1d(np.arange(N), walk)
			possible_negative_sample_frequencies = frequencies[possible_negative_samples]
			possible_negative_sample_frequencies /= possible_negative_sample_frequencies.sum()
			# negative_samples = np.random.choice(possible_negative_samples, replace=True, size=num_negative_samples, 
			# 				p=possible_negative_sample_frequencies / possible_negative_sample_frequencies.sum())

			for i in range(len(walk)):
				for j in range(i+1, min(len(walk), i+1+context_size)):
					if walk[i] == walk[j]:
						continue
					pair = np.array([walk[i], walk[j]])
					for negative_samples in np.random.choice(possible_negative_samples, replace=True, 
						size=(1+num_positive_samples, num_negative_samples),
						p=possible_negative_sample_frequencies):
						yield np.append(pair, negative_samples)
						pair = pair[::-1]

def grouper(n, iterable, fillvalue=None):
	'''grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx'''
	args = [iter(iterable)] * n
	return izip_longest(fillvalue=fillvalue, *args)