import numpy as np
import networkx as nx

from utils import compute_label_mask, generate_samples_node2vec, grouper, create_neighbourhood_sample_list

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