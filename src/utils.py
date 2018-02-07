import numpy as np
import networkx as nx
import scipy as sp

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from itertools import izip_longest

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from node2vec_sampling import Graph

def load_karate():

	G = nx.karate_club_graph()
	G = nx.convert_node_labels_to_integers(G)

	nx.set_edge_attributes(G, name="weight", values=1)

	X = sp.sparse.identity(len(G))

	map_ = {"Mr. Hi" : 0, "Officer" : 1}

	Y = np.zeros((len(G), 2))
	assignments = dict(nx.get_node_attributes(G, "club")).values()
	assignments =[map_[x] for x in assignments]
	Y[np.arange(len(G)), assignments] = 1
	Y = sp.sparse.csr_matrix(Y)


	X = X.toarray()
	Y = Y.toarray()

	return G, X, Y

def load_cora():

	G = nx.read_edgelist("../data/cora/cites.tsv", delimiter="\t", )
	G = nx.convert_node_labels_to_integers(G)
	
	nx.set_edge_attributes(G, name="weight", values=1)

	X = sp.sparse.load_npz("../data/cora/cited_words.npz")
	Y = sp.sparse.load_npz("../data/cora/paper_labels.npz")

	X = X.toarray()
	Y = Y.toarray()

	return G, X, Y

def load_facebook():

	G = nx.read_gml("../data/facebook/facebook_graph.gml",  )
	G = nx.convert_node_labels_to_integers(G)
	
	nx.set_edge_attributes(G, name="weight", values=1)

	X = sp.sparse.load_npz("../data/facebook/features.npz")
	Y = sp.sparse.load_npz("../data/facebook/circle_labels.npz")

	X = X.toarray()
	Y = Y.toarray()

	return G, X, Y

def preprocess_data(X):
	X = VarianceThreshold().fit_transform(X)
	X = StandardScaler().fit_transform(X)
	return X

def connect_layers(layer_tuples, x):
	
	y = x

	for layer_tuple in layer_tuples:
		for layer in layer_tuple:
			y = layer(y)
	return y



# def compute_negative_sampling_mask(batch_size, n, spacing):
# 	mask = np.zeros((batch_size, n*spacing))
# 	mask[:, np.arange(n) * spacing] = 1. / n
# 	return mask

def compute_label_mask(Y, num_patterns_to_keep=20):

	assignments = Y.argmax(axis=1)
	patterns_to_keep = np.concatenate([np.random.choice(np.where(assignments==i)[0], replace=False, size=num_patterns_to_keep)  
										 for i in range(Y.shape[1])])
	mask = np.zeros(Y.shape, dtype=np.float32)
	mask[patterns_to_keep] = 1

	return mask

def generate_samples_node2vec(G, num_positive_samples, num_negative_samples, context_size,
	p, q, num_walks, walk_length):
	
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

def create_neighbourhood_sample_list(nodes, neighbourhood_sample_sizes, neighbours):

	neighbourhood_sample_list = [nodes]

	for neighbourhood_sample_size in neighbourhood_sample_sizes[::-1]:
			neighbourhood_sample_list.append(
				np.array([
				np.concatenate([ 
					np.append(n, np.random.choice(neighbours[n], replace=True, size=neighbourhood_sample_size)) 
								for n in batch]) 
				for batch in neighbourhood_sample_list[-1]]))

	# flip neighbour list
	neighbourhood_sample_list = neighbourhood_sample_list[::-1]


	return neighbourhood_sample_list



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

	# X = X.toarray()
	# Y = Y.toarray()

	num_classes = Y.shape[1]
	if num_samples_per_class is not None:
		label_mask = compute_label_mask(Y, num_patterns_to_keep=num_samples_per_class)
	else:
		label_mask = np.ones(Y.shape)

	label_prediction_layers = np.where(num_capsules_per_layer==num_classes)[0] + 1

	
	# N = len(G)
	
	# node2vec_graph = Graph(nx_G=G, is_directed=False, p=p, q=q)
	# node2vec_graph.preprocess_transition_probs()
	# walks = node2vec_graph.simulate_walks(num_walks=num_walks, walk_length=walk_length)
	
	neighbours = [list(G.neighbors(n)) for n in list(G.nodes())]
	
	# frequencies = np.array(dict(G.degree()).values()) ** 0.75

	num_layers = neighbourhood_sample_sizes.shape[0]

	node2vec_sampler = generate_samples_node2vec(G, num_positive_samples, num_negative_samples, context_size, 
		p, q, num_walks, walk_length)
	batch_sampler = grouper(batch_size, node2vec_sampler)
	
	'''
	END OF PRECOMPUTATION
	'''
	
	# i = 0
	# nodes = np.random.permutation(N)
	# output_dimension = 1 + num_positive_samples + num_negative_samples
	# batch_nodes = np.zeros((batch_size, output_dimension), dtype=int)
	
	while True:

		batch_nodes = batch_sampler.next()
		batch_nodes = np.array(batch_nodes)

		# print batch_nodes

		# for j in range(batch_size):
			
		# 	if i == N:
		# 		nodes = np.random.permutation(N)
		# 		i = 0
				
		# 	n = nodes[i]
			
		# 	positive_samples = np.random.choice(walks[n], replace=True, size=num_positive_samples)
		# 	possible_negative_samples = np.setdiff1d(range(N), walks[n])
		# 	possible_negative_sample_frequencies = frequencies[possible_negative_samples]
		# 	negative_samples = np.random.choice(possible_negative_samples, replace=True, size=num_negative_samples, 
		# 					p=possible_negative_sample_frequencies / possible_negative_sample_frequencies.sum())
													 
		# 	batch_nodes[j, 0] = n
		# 	batch_nodes[j, 1 : 1 + num_positive_samples] = positive_samples
		# 	batch_nodes[j, 1 + num_positive_samples : ] = negative_samples
		
		# 	i += 1
	
		# neighbour_list = [batch_nodes]
		# for neighbourhood_sample_size in neighbourhood_sample_sizes[::-1]:
		# 	neighbour_list.append(np.array([
		# 		np.concatenate([ np.append(n, np.random.choice(neighbours[n], replace=True, size=neighbourhood_sample_size)) 
		# 						for n in batch]) for batch in neighbour_list[-1]]))

		# # flip neighbour list
		# neighbour_list = neighbour_list[::-1]

		neighbour_list = create_neighbourhood_sample_list(batch_nodes, neighbourhood_sample_sizes, neighbours)

		# print neighbour_list

		# shape is [batch_size, output_shape*prod(sample_sizes), D]
		x = X[neighbour_list[0]]
		x = np.expand_dims(x, 2)
		# shape is now [batch_nodes, output_shape*prod(sample_sizes), 1, D]


		negative_sample_targets = [Y[nl].argmax(axis=-1) for nl in neighbour_list[1:]]

		# print label_prediction_layers

		labels = []
		for layer in label_prediction_layers:
			y = Y[neighbour_list[layer]]
			mask = label_mask[neighbour_list[layer]]
			y_masked = np.append(mask, y, axis=-1)
			labels.append(y_masked)

		# print len(labels)
		# print len(negative_sample_targets)

		if all([(y_masked[:,:,:num_classes] > 0).any() for y_masked in labels]):
			yield x, labels + negative_sample_targets

		# y_label_mask = label_mask[neighbour_list[1]]
		# if (y_label_mask>0).any():
		# 	y = Y[neighbour_list[1]]
		# 	y_label_true = np.append(np.ones(y.shape), y, axis=-1)
		# 	y_label_mask = np.append(y_label_mask, y, axis=-1)

			

		# 	yield x, [y_label_true] + [y_label_mask] + negative_sample_targets
			# yield x, negative_sample_targets

def plot_embedding(G, X, Y, embedder, neighbourhood_sample_sizes, batch_size, dim=2, annotate=False, path=None):

	# x, yl = generator.next()
	# x, [_, y, _, _] = generator.next()

	print "Plotting network and saving to {}...".format(path)

	# y = yl[-1]
	nodes = np.arange(len(G)).reshape(-1, 1)
	neighbours = [list(G.neighbors(n)) for n in list(G.nodes())]
	neighbour_list = create_neighbourhood_sample_list(nodes, neighbourhood_sample_sizes, neighbours)

	x = X[neighbour_list[0]]
	# print x.shape
	x = np.expand_dims(x, 2)
	# print x.shape

	embedding = embedder.predict(x, batch_size=batch_size)
	# print embedding.shape
	embedding = embedding.reshape(-1, dim)
	# print embedding.shape


	y = Y.argmax(axis=1)

	fig = plt.figure(figsize=(5, 5))
	if dim == 3:
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2], c=y)
	else:
		plt.scatter(embedding[:,0], embedding[:,1], c=y)
		if annotate:
			for label, p in zip(list(G), embedding[:,:2]):
				plt.annotate(label, p)
	# plt.show()
	if path is not None:
		plt.savefig(path)

	print "Done"
	

